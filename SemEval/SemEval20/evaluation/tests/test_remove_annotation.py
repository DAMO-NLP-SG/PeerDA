import sys
sys.path.append("../")
import src.propaganda_techniques as pt
import src.annotation as an
import src.article_annotations as aa

def test_remove_annotation(artannotations):
    before = str(artannotations)
    print("removing annotation: " + str(artannotations[0]))
    artannotations.remove_annotation(artannotations[0])    
    after = str(artannotations)
    assert after==before.replace("\n\t[0, 59] -> Exaggeration,Minimisation","",1)


if __name__ == "__main__":

    propaganda_techniques = pt.Propaganda_Techniques(filename="../data/propaganda-techniques-names.txt")
    an.Annotation.set_propaganda_technique_list_obj(propaganda_techniques)

    artannotations = aa.Articles_annotations()
    artannotations.load_article_annotations_from_csv_file("../data/article736757214.task-FLC.labels")
    test_remove_annotation(artannotations)
