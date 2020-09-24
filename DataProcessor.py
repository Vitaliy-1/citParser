import os
from xml.etree import ElementTree
from keras.preprocessing.text import text_to_word_sequence

class DataProcessor:
    listOfTuples = []
    words = []
    tags = []
    maxLength = 0
    allTags = []
    allWords = []

    def __init__(self, pathToDataDir, type):
        self.pathToDataDir = pathToDataDir
        self.type = type
        if (type == 'tei'):
            self.__processTei()

        self.words = list(set(self.allWords))
        self.tags = list(set(self.allTags))

    def __processTei(self):
        for filename in os.listdir(self.pathToDataDir):
            fullname = os.path.join(self.pathToDataDir, filename)
            tree = ElementTree.parse(fullname)
            for bibl in tree.iter('{http://www.tei-c.org/ns/1.0}bibl'):
                item = []
                for anyNode in list(bibl):
                    if anyNode.tag == 'author' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}author':
                        self.__tagsToTuples(item, anyNode, 'author')
                    elif anyNode.tag == 'date' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}date':
                        self.__tagsToTuples(item, anyNode, 'date')
                    elif anyNode.tag == 'biblScope' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}biblScope':
                        attrib = anyNode.attrib
                        if attrib.get('unit') == 'volume' or attrib.get('type') == 'vol':
                            self.__tagsToTuples(item, anyNode, 'volume')
                        elif attrib.get('unit') == 'page' or attrib.get('type') == 'pp':
                            self.__tagsToTuples(item, anyNode, 'page')
                    elif anyNode.tag == 'title' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}title':
                        attrib = anyNode.attrib
                        if attrib.get('level') == 'j':
                            self.__tagsToTuples(item, anyNode, 'journal')
                        elif attrib.get('level') == 'a':
                            self.__tagsToTuples(item, anyNode, 'title')
                        elif attrib.get('level') == 'm':
                            self.__tagsToTuples(item, anyNode, 'container_title')
                    elif anyNode.tag == 'idno' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}idno':
                        attrib = anyNode.attrib
                        if attrib.get('type') == 'arxiv':
                            self.__tagsToTuples(item, anyNode, 'id_arxiv')
                        elif attrib.get('type') == 'report':
                            self.__tagsToTuples(item, anyNode, 'id_report')
                        elif attrib.get('type') and attrib.get('type').lower() == 'issn':
                            self.__tagsToTuples(item, anyNode, 'id_issn')
                        elif attrib.get('type') and attrib.get('type').lower() == 'isbn':
                            self.__tagsToTuples(item, anyNode, 'id_isbn')
                        elif attrib.get('type') and attrib.get('type').lower() == 'doi':
                            self.__tagsToTuples(item, anyNode, 'id_doi')
                        elif attrib.get('type') and attrib.get('type').lower() == 'pmid':
                            self.__tagsToTuples(item, anyNode, 'id_pmid')
                        elif attrib.get('type') and attrib.get('type').lower() == 'pmc':
                            self.__tagsToTuples(item, anyNode, 'id_pmc')
                        else:
                            self.__tagsToTuples(item, anyNode, 'id_other')
                    elif anyNode.tag == 'publisher' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}publisher':
                        self.__tagsToTuples(item, anyNode, 'publisher')
                    elif anyNode.tag == 'orgName' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}orgName':
                        self.__tagsToTuples(item, anyNode, 'publisher')
                    elif anyNode.tag == 'pubPlace' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}pubPlace':
                        self.__tagsToTuples(item, anyNode, 'publisher')
                    elif anyNode.tag == 'ptr' or anyNode.tag == '{http://www.tei-c.org/ns/1.0}ptr':
                        attrib = anyNode.attrib
                        if attrib.get('type') == 'web':
                            self.__tagsToTuples(item, anyNode, 'url')

                self.listOfTuples.append(item)

                itemLength = len(item)
                if (self.maxLength < itemLength):
                    self.maxLength = itemLength


    def __tagsToTuples(self, item, tag, tagName=None):
        if tagName is None:
            tagName = tag.tag

        text = ''.join(tag.itertext())
        tokenList = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=False)

        for index, token in enumerate(tokenList):
            """
            if (index == 0):
                tokenLabel = 'I-' + tagName
            else:
                tokenLabel = tagName

            # Add label for the vocab
            if (tokenLabel not in self.tags):
                self.tags.append(tokenLabel)
            # Add word for the vocab
            if (token not in self.words):
                self.words.append(token)
            """
            tokenLabel = tagName
            self.allTags.append(tokenLabel)
            self.allWords.append(token)
            item.append((token, tokenLabel))

    def getListOfTuples(self):
        return self.listOfTuples

    def getTags(self):
        return self.tags

    def getWords(self):
        return self.words

    def getMaxLength(self):
        return self.maxLength

