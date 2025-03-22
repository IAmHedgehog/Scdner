import spacy
from spacy.tokens import Doc
from tqdm import tqdm

parser = spacy.load('en_core_web_trf')


def parse_data(data: dict) -> list:
    words = data['tokens']
    doc = Doc(parser.vocab, words=words)
    tokens = parser(doc)
    return tokens


def clean_data(dataset: list) -> list:
    # remove mis-annotated entity with PUNCT as ending word
    # such as [Dec, 12, .] should be [Dec, 12] with . removed
    for idx, cur_d in enumerate(tqdm(dataset)):
        words = cur_d['tokens']
        doc = Doc(parser.vocab, words=words)
        tokens = parser(doc)
        for ent in cur_d['entities']:
            ent_words = cur_d['tokens'][ent['start']: ent['end']]
            if len(ent_words[-1]) == 1 and tokens[ent['end'] - 1].pos_ == 'PUNCT':
                ent['end'] -= 1
                print('fixed ----post---->', idx, ent_words)
            if len(ent_words[0]) == 1 and tokens[ent['start']].pos_ == 'PUNCT':
                ent['start'] += 1
                print('fixed ----pre----->', idx, ent_words)

        new_words = []
        for idx, word in enumerate(words):
            if len(word) > 1 and word[0] == ',':
                new_words.append(',')
                new_words.append(word[1:].strip())
                print('fixed -----starting comma------->', idx, word)
                for ent in cur_d['entities']:
                    if ent['start'] >= idx:
                        print('move ent start from ', ent['start'], ' to ', ent['start'] + 1)
                        ent['start'] += 1
                    if ent['end'] >= idx:
                        print('move ent end from ', ent['end'], ' to ', ent['end'] + 1)
                        ent['end'] += 1
            else:
                new_words.append(word)
        cur_d['tokens'] = new_words
    return dataset


def get_reachable_tokens(tokens: list):
    token_dic = {}
    post_token_dic = {}
    token_idx_dic = {}
    for idx, token in enumerate(tokens):
        token_idx_dic[token] = idx
        token_dic[token] = [token]
        post_token_dic[token] = [token]
    for token in tokens:
        cur_token = token
        while cur_token.head != cur_token:
            if token_idx_dic[token] < token_idx_dic[cur_token.head]:
                # token can be reached by heads
                # this if ensure only previous words are reached
                token_dic[cur_token.head].append(token)
            else:
                post_token_dic[cur_token.head].append(token)
            cur_token = cur_token.head

    for key, tokens in token_dic.items():
        token_dic[key] = sorted(set(tokens), key=lambda token: token_idx_dic[token], reverse=True)
    for key, tokens in post_token_dic.items():
        post_token_dic[key] = sorted(set(tokens), key=lambda token: token_idx_dic[token])
    return token_dic, post_token_dic, token_idx_dic


def is_valid_ent_end_token(token):
    # return is_valid, is_expandable (if it could be included in spans with multiple words)
    if token.pos_ in ['PROPN', 'NUM', 'NOUN']:
        return True, True
    if token.pos_ == 'ADJ' and token.text[0].isupper():
        # ADJ entities are alway single word
        return True, False
    return False, False


def is_valid_ent_start_token(token):
    # return is_valid
    if token.pos_ in ['PROPN', 'NUM', 'SYM']:
        return True
    if token.pos_ == 'NOUN':
        return True
    if token.pos_ == 'ADJ':
        return True
    if token.pos_ == 'DET' and token.text[0].isupper() and token.text.lower() == 'the':
        return True
    return False

def is_valid_ent_start_token_v1(token):
    # works well in general domaind datasets such as CONLL
    # return is_valid
    if token.pos_ in ['PROPN', 'NUM', 'SYM']:
        return True
    if token.pos_ == 'NOUN' and token.text[0].isupper():
        return True
    if token.pos_ == 'ADJ' and token.text[0].isupper():
        return True
    if token.pos_ == 'DET' and token.text[0].isupper() and token.text.lower() == 'the':
        return True
    return False


def is_valid_punct_span(span):
    VALID_PUNCTS = [',', '(', ')']
    left_par_cnt = 0
    right_par_cnt = 0
    for token in span:
        if token.pos_ == 'PUNCT' and token.text not in VALID_PUNCTS:
            return False
        if token.text == '(':
            left_par_cnt += 1
        if token.text == ')':
            right_par_cnt += 1
    if left_par_cnt != right_par_cnt:
        return False
    return True


def is_valid_span(span):
    MAX_SPAN_LEN = 10
    if not is_valid_ent_start_token(span[0]):
        return False

    if len(span) > MAX_SPAN_LEN:
        return False

    if not is_valid_punct_span(span):
        return False

    return True


def get_ent_spans(tokens: list) -> set:
    candidates = []
    token_dic, post_token_dic, token_idx_dic = get_reachable_tokens(tokens)
    for token in tokens:
        # generate spans that only include words before token
        candidates.extend(get_token_spans(token, token_dic, token_idx_dic))
        # generate spans that contains words after token
        candidates.extend(get_post_token_spans(token, post_token_dic[token], token_dic[token], token_idx_dic))

    candidates.extend(get_date_spans(tokens))
    candidates.extend(get_loc_spans(tokens))
    # convert to (span_start, span_end) format

    candidates = [can for can in candidates if is_valid_span(can)]

    candidates = [(token_idx_dic[can[0]], token_idx_dic[can[-1]]+1) for can in candidates]
    return set(candidates)


def get_post_token_spans(token, post_tokens, pre_tokens, token_idx_dic):
    # try to make it work on sci domain datasets such as scierc
    # if token.pos_ != 'PROPN':
        # only PROPN followed by ADP can generate valid spans
        # return []
    
    filtered_post_tokens = []
    
    for idx, p_token in enumerate(post_tokens):
        if token_idx_dic[p_token] - token_idx_dic[token] != idx:
            break
        filtered_post_tokens.append(p_token)

    # if len(filtered_post_tokens) < 3 or filtered_post_tokens[1].pos_ != 'ADP':
        # only PROPN followed by ADP can generate valid spans
        # should be at least "A of B", where of can be other ADP words
        # return []

    candidates = [filtered_post_tokens]
    # cur_ent = post_tokens[:2]
    # could be multiple ADP, segment each of them
    for idx in range(2, len(filtered_post_tokens)):
        if filtered_post_tokens[idx].pos_ != 'PROPN':
            candidates.append(filtered_post_tokens[:idx])

    filter_pre_tokens = []
    for idx in range(1, len(pre_tokens)):
        if pre_tokens[idx].pos_ != 'PROPN':
            break
        filter_pre_tokens = [pre_tokens[idx]] + filter_pre_tokens

    candidates = [filter_pre_tokens + can for can in candidates if can[-1].pos_ in ['PROPN', 'NOUN']]

    return candidates


def get_post_token_spans_v1(token, post_tokens, pre_tokens, token_idx_dic):
    # works well for general domain datasets such as CONLL
    if token.pos_ != 'PROPN':
        # only PROPN followed by ADP can generate valid spans
        return []
    
    filtered_post_tokens = []
    
    for idx, p_token in enumerate(post_tokens):
        if token_idx_dic[p_token] -  token_idx_dic[token] != idx:
            break
        filtered_post_tokens.append(p_token)
    
    if len(filtered_post_tokens) < 3 or filtered_post_tokens[1].pos_ != 'ADP':
        # only PROPN followed by ADP can generate valid spans
        # should be at least "A of B", where of can be other ADP words
        return []
    
    candidates = [filtered_post_tokens]
    # cur_ent = post_tokens[:2]
    # could be multiple ADP, segment each of them
    for idx in range(2, len(filtered_post_tokens)):
        if filtered_post_tokens[idx].pos_ != 'PROPN':
            candidates.append(filtered_post_tokens[:idx])
    
    filter_pre_tokens = []
    for idx in range(1, len(pre_tokens)):
        if pre_tokens[idx].pos_ != 'PROPN':
            break
        filter_pre_tokens = [pre_tokens[idx]] + filter_pre_tokens
    
    candidates = [filter_pre_tokens + can for can in candidates]

    return candidates


def get_token_spans(token, token_dic, token_idx_dic):
    candidates = []
    is_ent, expandable = is_valid_ent_end_token(token)
    if is_ent:
        candidates.append([token])
    if expandable:
        cur_ent = [token]
        cur_idx = 1
        while cur_idx < len(token_dic[token]):
            if cur_idx < len(token_dic[token]) and token_idx_dic[token_dic[token][cur_idx]] + 1 == token_idx_dic[cur_ent[0]]:
                candidates.append([token_dic[token][cur_idx]] + cur_ent)
                cur_ent = [token_dic[token][cur_idx]] + cur_ent
                cur_idx += 1
            else:
                break
    return candidates


def get_date_spans(tokens: list) -> list:
    candidates = []
    poss = [token.pos_ for token in tokens]
    for idx in range(1, len(tokens)-1):
        if poss[idx-1: idx+2] == ['NUM', 'PROPN', 'NUM']:
            # Day, month, year
            candidates.append(tokens[idx-1: idx+2])
        elif idx < len(tokens) -2 and poss[idx-1: idx+3] == ['PROPN', 'NUM', 'PUNCT', 'NUM']:
            # Month, Day, comma, year
            candidates.append(tokens[idx-1: idx+3])
        elif poss[idx: idx+2] ==  ['PROPN', 'NUM'] and tokens[idx+1].head == tokens[idx]:
            # Month, day
            candidates.append(tokens[idx: idx+2])
    return candidates


def get_loc_spans(tokens: list) -> list:
    candidates = []
    poss = [token.pos_ for token in tokens]
    for idx in range(len(tokens)-2):
        if poss[idx: idx+3] == ['PROPN', 'PUNCT', 'PROPN'] and tokens[idx+2].dep_ == 'appos':
            # City, comma, State
            candidates.append(tokens[idx: idx+3])
        if idx < len(tokens) - 3 and poss[idx: idx+4] == ['PROPN', 'PROPN', 'PUNCT', 'PROPN'] and tokens[idx+3].dep_ == 'appos':
            # City1, City2, comma, State
            candidates.append(tokens[idx: idx+4])
    return candidates
