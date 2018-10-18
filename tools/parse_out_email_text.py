#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import nltk

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    stemmer = SnowballStemmer("english")
    words_stem_list = []
    if len(content) > 1:
        ### remove punctuation
        # text_string = content[1].translate(string.maketrans("", ""), string.punctuation) # Python 2.7
        text_string = content[1].translate(str.maketrans("", ""))  # PYthon 3

        ### project part 2: comment out the line below
        #words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words_str = ""
        # words_full = str.split(text_string) # Expected by Udacity
        words_full = nltk.word_tokenize(text_string,) # I like this more!
        for word_full in words_full:
            #words_stem_list.append(stemmer.stem(word_full))
            words_str += stemmer.stem(word_full)
            words_str += " "
            #words_stem_list.append(word_full)

    #words_str = ""
    #for element in words_stem_list:
    #    if not("." in element or "!" in element or "," in element or "?" in element): # or element.find(",") or element.find("?")):
    #        words_str += " "
    #    words_str += str(element)

    return words_str



def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()
