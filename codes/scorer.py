from nltk.translate.bleu_score import sentence_bleu as get_bleu, corpus_bleu

output_dir = "model_outputs/nopunc/test.hi"
ref_dir = "model_outputs/nopunc/en_hi_outputs.txt"

# fref = open("test_raw.hi", 'r')
refs = open(ref_dir, 'r').readlines()
print(f"refs = {len(refs)}")
outs = open(output_dir, 'r').readlines()
print(f"outs = {len(outs)}")

if len(refs) != len(outs):
    print("File length mismatch!! Ensure Refernce and Output have same number of lines")
    exit(0)
ref_tokenized = [ref.strip().split(' ') for ref in refs]
out_tokenized = [out.strip().split(' ') for out in outs]

ref_corpus = [[ref] for ref in ref_tokenized ]

score = corpus_bleu(ref_corpus, out_tokenized)

print(f"corpus blue:\t{score}")
