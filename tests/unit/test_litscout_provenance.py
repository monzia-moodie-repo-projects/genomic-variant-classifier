"""test_litscout_provenance.py  --  Monzia Moodie

Provenance capture in LiteratureScout: the PubMed efetch XML parse
(journal / authors / publication_date) with its date-format fallbacks.
"""
import xml.etree.ElementTree as ET

from genomic_variant_classifier.agent_layer.agents.literature_scout_agent import (
    _parse_pubmed_article,
)

_XML = """<PubmedArticleSet>
 <PubmedArticle><MedlineCitation>
   <PMID>40123456</PMID>
   <Article>
     <Journal><Title>Nature Methods</Title><JournalIssue>
       <PubDate><Year>2025</Year><Month>03</Month></PubDate></JournalIssue></Journal>
     <ArticleTitle>A graph neural network for variant effect prediction</ArticleTitle>
     <Abstract><AbstractText Label="BACKGROUND">We present GraphVEP.</AbstractText>
       <AbstractText Label="RESULTS">It improves AUROC.</AbstractText></Abstract>
     <AuthorList>
       <Author><LastName>Moodie</LastName><Initials>M</Initials></Author>
       <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
       <Author><CollectiveName>The GenAssoc Consortium</CollectiveName></Author>
     </AuthorList>
     <ArticleDate><Year>2025</Year><Month>3</Month><Day>7</Day></ArticleDate>
   </Article>
 </MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation>
   <PMID>40999999</PMID>
   <Article>
     <Journal><ISOAbbreviation>Bioinformatics</ISOAbbreviation><JournalIssue>
       <PubDate><MedlineDate>2024 Winter</MedlineDate></PubDate></JournalIssue></Journal>
     <ArticleTitle>Protein language model scoring</ArticleTitle>
     <Abstract><AbstractText>No structured date here.</AbstractText></Abstract>
     <AuthorList><Author><LastName>Lee</LastName><Initials>K</Initials></Author></AuthorList>
   </Article>
 </MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def _arts():
    return ET.fromstring(_XML).findall(".//PubmedArticle")


def test_pubmed_journal_authors_date_full():
    a = _parse_pubmed_article(_arts()[0])
    assert a["pmid"] == "40123456"
    assert a["journal"] == "Nature Methods"
    assert a["authors"] == "Moodie M; Smith J; The GenAssoc Consortium"
    assert a["publication_date"] == "2025-03-07"          # ArticleDate, zero-padded
    assert a["abstract"] == "We present GraphVEP. It improves AUROC."  # multi-AbstractText


def test_pubmed_fallbacks():
    a = _parse_pubmed_article(_arts()[1])
    assert a["journal"] == "Bioinformatics"               # ISOAbbreviation fallback
    assert a["publication_date"] == "2024 Winter"         # MedlineDate fallback
    assert a["authors"] == "Lee K"


def test_pubmed_missing_fields_safe():
    art = ET.fromstring(
        "<PubmedArticle><MedlineCitation><PMID>1</PMID>"
        "<Article><ArticleTitle>t</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
    )
    p = _parse_pubmed_article(art)
    assert p["pmid"] == "1"
    assert p["journal"] == "" and p["authors"] == "" and p["publication_date"] == ""
    assert p["source"] == "PubMed"
