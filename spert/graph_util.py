import spacy
from spacy.tokens import Doc
import networkx as nx
import numpy as np

parser = spacy.load('en_core_web_trf')


def gen_graph(doc_tokens):
    words = [dt.phrase for dt in doc_tokens]
    doc = Doc(parser.vocab, words=words)
    tokens = parser(doc)

    g = nx.Graph()
    dic = {}
    g.add_node(0, text='START')
    for tid, token in enumerate(tokens):
        for span in range(doc_tokens[tid].span_start, doc_tokens[tid].span_end):
            g.add_node(
                span, text=token.text, tag=token.tag_,
                pos=token.pos_, dep=token.dep_)
            if span > doc_tokens[tid].span_start:
                g.add_edge(span, span-1, dep=token.dep_)
        dic[token] = tid
    g.add_node(len(g.nodes()), text='END')

    for token in tokens:
        if token != token.head:
            for span1 in range(doc_tokens[dic[token]].span_start, doc_tokens[dic[token]].span_end):
                for span2 in range(doc_tokens[dic[token.head]].span_start, doc_tokens[dic[token.head]].span_end):
                    g.add_edge(span1, span2, dep=token.dep_)
    g.add_edge(0, 1, dep='start')
    g.add_edge(len(g.nodes())-1, len(g.nodes())-2, dep='end')

    return nx.adjacency_matrix(g).todense() + np.eye(len(g.nodes()))
