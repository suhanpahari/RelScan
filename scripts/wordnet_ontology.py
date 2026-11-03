"""WordNet neighbourhoods of every label (run with a Python that has nltk; NLTK_DATA must point at the corpus).
  syn[c]  labels sharing a noun synset with c                (synonyms: sofa ~ couch)
  hyp[c]  labels whose noun synsets are c's or below them     (is-a c: armchair, office chair -> chair)
Out-of-WordNet labels (multi-word compounds WordNet lacks) get the compound's head noun as a fallback parent
("dining table" -> table), so "chair" also covers "sofa chair" and "folded chair"."""
import json
import sys

from nltk.corpus import wordnet as wn


def synsets(w):
    return wn.synsets(w.replace(" ", "_"), pos=wn.NOUN)


def main():
    labels = sorted({l for f in sys.argv[2:] for l in json.load(open(f))})
    S = {w: set(synsets(w)) for w in labels}
    down = {w: set().union(*[set(s.closure(lambda x: x.hyponyms())) | {s} for s in S[w]]) if S[w] else set() for w in labels}
    syn, hyp, nown = {}, {}, []
    for c in labels:
        syn[c] = sorted(w for w in labels if w == c or (S[c] & S[w]))
        hyp[c] = sorted(w for w in labels if w == c or (S[w] and S[w] <= down[c]) or (S[w] & down[c] and not S[w] - down[c] - S[c])
                        or (not S[w] and " " in w and w.split()[-1] == c))
        if not S[c]:
            nown.append(c)
    json.dump(dict(syn=syn, hyp=hyp, not_in_wordnet=nown), open(sys.argv[1], "w"), indent=0)
    for c in ("chair", "table", "sofa", "bin", "lamp", "pillow", "door", "cabinet", "trash can", "light"):
        if c in labels:
            print(c, "| syn:", [w for w in syn[c] if w != c], "| hyp:", [w for w in hyp[c] if w != c][:14])
    print(len(labels), "labels,", len(nown), "not in WordNet:", nown[:40])


if __name__ == "__main__":
    main()
