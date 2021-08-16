# Adapted from https://github.com/shahidost/Baseline4VTKEL.git
# Modified by:
# Author name: Yueqing Xuan
# Student ID: 1075355

# PRE-REQUISITES
#   pip3 install rdflib

import requests
from rdflib import Graph, URIRef, Literal, Namespace, ConjunctiveGraph

# connect to 'PIKES server' for knowledge graph in RDF Trig format
PUBLIC_PIKES_SERVER = 'https://knowledgestore2.fbk.eu/pikes-demo/api/'
LOCAL_PIKES_SERVER = 'http://localhost:8011/'

g1 = Graph()


def pikes_text2rdf(sentence):
    """
    This function takes a natural language sentence and passed through ‘PIKES server’ for knowledge graph extraction
    input:
      img_caption – input natural language text

    output:
      .ttl file – a turtle RDF format output file,  which stored the knowledge graph of natural language in Triples form
    """
    return requests.get(PUBLIC_PIKES_SERVER + "text2rdf?", {'text': sentence}, verify=False)


def get_entities_from_text(text):
    """
    This function extract RDF graph for textual entities, their YAGO type, beginindex and endindex, processed by PIKES tool.
    input:
      text –  a sentence

    output:
     sparql query result  – stored textual entities recognized and linked by PIKES
    """

    pikes_answer = pikes_text2rdf(text.lower())

    g = ConjunctiveGraph()
    g.parse(data=pikes_answer.content.decode('utf-8'), format="trig")
    # Sparql query for entities information extraction
    sparql_query = """SELECT ?TED ?TEM ?TET ?anchor ?beginindex ?endindex
           WHERE {
           GRAPH ?g1 {?TED <http://groundedannotationframework.org/gaf#denotedBy> ?TEM}
           GRAPH ?g2 {?TED a ?TET}
           GRAPH ?g3 {?TEM nif:anchorOf ?anchor}
           GRAPH ?g4 {?TEM nif:beginIndex ?beginindex}
           GRAPH ?g5 {?TEM nif:endIndex ?endindex}
           }"""

    return g.query(sparql_query)


def Textual_entities_detection_linking(sentence):
    print('\n-------------------------------------------------\nPIKES entities processing....')
    """
    This function detects and links textual entities in a sentence using the PIKES tool and saves it into .ttl file.  
    input:
        sentence (a RDF object) - the text where the textual entities will be detected
    output:
        textual_entities - Textual entities detected by PIKES
        textual_entities_YAGO_type - Textual entities YAGO types linked by PIKES
    """

    # TED and TET using PIKES
    textual_entities = []
    textual_entities_YAGO_type = []
    textual_entity_mentions = []

    """
    attributes of an instance in textual_entities_index
    
    attribute 0: textual entity
    attribute 2: class of the entity in ontology
    attribute 3: textual mention of entity
    attribute 4: begin index
    attribute 5: end index
    """
    textual_entities_index = get_entities_from_text(sentence)

    for row1 in textual_entities_index:

        if 'http://dbpedia.org/class/yago/' in row1[2] \
                and 'http://www.newsreader-project.eu/time/P1D' not in row1[0]:
            uri_textual_entity = URIRef(row1[0][21:])

            textual_entities.append(row1[0][21:])
            textual_entities_YAGO_type.append(row1[2][30:])
            textual_entity_mentions.append(row1[3].n3())

    return textual_entities, textual_entity_mentions, textual_entities_YAGO_type


if __name__ == "__main__":
    command = "Move the purple pillows from the couch to the armchair and walk around the table"

    textual_entities, textual_mentions, textual_entities_YAGO_type = Textual_entities_detection_linking(command)
    print("\ntextual mentions of entities: ", textual_mentions)
    print("textual entities: ", textual_entities)
    print("YAGO class of entities: ", textual_entities_YAGO_type)
    # g1.serialize(format="turtle")
