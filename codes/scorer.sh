# change values of REF and PRED to reference file and prediction file respectively to calculate score.
# if Sacrebleu module is not installed, install it using:
#       python3 -m pip install sacrebleu

REF="ref.txt"
PRED="pred.txt"

sacrebleu $REF -i $PRED -m bleu -b -w 4