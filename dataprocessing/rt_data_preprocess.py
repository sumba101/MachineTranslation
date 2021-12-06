# This file was used to combine the provided RT annotated files into a single dataset split into train-test-dev,
# as well as process it to obtain the english sentences without annotations and without punctuations for testing with different models

# To run, it needs to be present in a directory with the following structure:
# <parent_dir>
#   - combined
#       - rt_data_preprocess.py
#       - en            //should contain all the files named as <name>.en.richtext1 from all folders in the RT dataset
#       - hi            //should contain all the files named as <name>.hi.translated from all folders in the RT dataset
#       - filtered      //dir where the output will be saved
#           - train
#           - dev
#           - test      //these three must preferably be empty.









# %%
import os
import unicodedata
import pandas as pd


# %%
file_list = os.listdir('en/')
# file_list

# %%
len(file_list)

# %%
prefixes = []
for file in file_list:
    file = file.split('.')[0]
    prefixes.append(file)

# %%
len(prefixes)

# %%
en_corp = []
hi_corp = []
pfile = open("mismatch.txt", 'w')
# prefixes = ['W2_AI_MOdule_6', 'W6_AI_MOdule_16']
for prefix in prefixes:
    print(prefix)
    en_lines = open(f"en/{prefix}.en.richtext1")
    hi_lines = open(f"hi/{prefix}.hi.translated")
    en_filt = []
    hi_filt = []
    for line in en_lines:
        if unicodedata.normalize('NFKD', line.strip()) == '' or unicodedata.normalize('NFKD', line.strip().lower()) == 'intentionally left blank':
            continue
        en_filt.append(unicodedata.normalize('NFKD', line.strip()))
    print(len(en_filt))
    for line in hi_lines:
        if unicodedata.normalize('NFKD', line.strip()) == '' or unicodedata.normalize('NFKD', line.strip().lower()) == 'intentionally left blank':
            continue
        hi_filt.append(unicodedata.normalize('NFKD', line.strip()))
    print(len(hi_filt))
    if len(hi_filt) != len(en_filt):
        # print(f"Pronlem at {prefix}")
        pfile.write(f"{prefix}\n")
    else:
        en_corp = en_corp + en_filt
        hi_corp = hi_corp + hi_filt
print(len(en_corp), len(hi_corp))        

# %%
def rem_annots(inp, sc, ec):
    st = inp.find(sc)
    new_st = 0
    # print(inp)
    while st >= 0:
        # print(st)
        if st < 0:
            return inp
        end = inp.find(ec, st)
        if end < 0:
            return inp
        # print(inp[st:end+2])
        to_rep = inp[st:end+2]
        temp = to_rep[2:-2]
        # print(temp, to_rep)
        toks = temp.split('|')
        # print(toks)
        if toks[0] == '':
            new_tok = toks[1]
        else:
            new_tok = toks[0]
        new_tok += ' '
        inp = inp.replace(to_rep, new_tok)
        new_st = inp.find(new_tok) + len(new_tok)
        # print(inp)
        # print()
        st = inp.find(sc, new_st)
    inp = unicodedata.normalize('NFKD', inp.replace("  ", " "))
        # print(inp[new_st:])
    return str(inp)

# %%
hi_corp_norm = []
for line in hi_corp:
    new_line = rem_annots(line, '((', ')')
    newer_line = rem_annots(new_line, '[[', ']')
    hi_corp_norm.append(newer_line)

# %%
comb_df = pd.DataFrame(
    {
        'en' : en_corp,
        'hi' : hi_corp_norm
    }
)
comb_df.sample(20)

# %%
comb_df['en'].astype(str)
comb_df['hi'].astype(str)

filt_df = comb_df[comb_df['en'].str.split().str.len() > 5]
filt_df
# filt_df.to_pickle('comb_df.pkl')

# %%
# parts = inp.split('((')
# parts

# %%
# inp = "((| पॉलीपेप्टाइडों| polypeptides)) की संरचना के अध्ययन के लिए, ((इम्पिरिकल स्थितिज ऊर्जा फलन| इम्पिरिकल पोटेंशियल एनर्जी फंक्शन| empirical potential energy function)) क्या होता है? एक ((| पॉलीपेप्टाइड| polypeptide)) की संरचना का अध्ययन, ((एब इनिशो क्वांटम मैकेनिकल अध्ययन| एब इनिशो क्वांटम मैकेनिकल स्टडीज़| ab initio quantum mechanical studies)), ((| सेमी इम्पिरिकल| semi empirical)), ((इम्पिरिकल क्वांटम रासायनिक विधियों| इम्पिरिकल क्वांटम केमिकल मेथड्स| empirical quantum chemical methods)) से शुरू करते हुए विशुद्ध ((इम्पिरिकल परमाणु परमाणु| इम्पिरिकल एट्म एट्म| empirical atom atom)) आधारित ((| एडिटिव नॉन बॉन्डिड पोटेंशियल| additive non bonded potential)) तक विभिन्न प्रकार के सैद्धांतिक तरीकों से किया जा सकता है। यह पूरी तरह से एक ((| पैरामीट्रिक पोटेंशियल| parametric potential) होता है जिसे संरचना ऊर्जा की गणना के लिए उपयोग किया जाता है।"

# st = inp.find('((')
# while st >= 0:
#     st = inp.find('((')
#     if st < 0:
#         break
#     end = inp.find(')', st)
#     # print(inp[st:end+2])
#     to_rep = inp[st:end+2]
#     temp = to_rep[2:-2]
#     # print(temp, to_rep)
#     toks = temp.split('|')
#     print(toks)
#     if toks[0] == '':
#         new_tok = toks[1]
#     else:
#         new_tok = toks[0]
#     new_tok += ' '
#     inp = inp.replace(to_rep, new_tok)
#     new_st = inp.find(new_tok) + len(new_tok)
#     print(inp)
#     print()
#     # print(inp[new_st:])
#     st = new_st

# %%
dev_test_df = filt_df.sample(frac = 0.20)


train_df = filt_df.drop(dev_test_df.index)
dev_df = dev_test_df.sample(frac = (0.50))
test_df = dev_test_df.drop(dev_df.index)
print(len(train_df), len(test_df), len(dev_df))




test_df['en'] = test_df['en'].apply(lambda x: unicodedata.normalize('NFKD', x))
test_df.to_csv("filtered/test/test.en.raw", sep=' ', columns=['en'], index=False, header=False, encoding='utf-8')
test_df['hi'] = test_df['hi'].apply(lambda x: unicodedata.normalize('NFKD', x))
test_df.to_csv("filtered/test/test.hi.raw", sep=' ', columns=['hi'], index=False, header=False, encoding='utf-8')

dev_df['en'] = dev_df['en'].apply(lambda x: unicodedata.normalize('NFKD', x))
dev_df.to_csv("filtered/dev/dev.en.raw", sep=' ', columns=['en'], index=False, header=False, encoding='utf-8')
dev_df['hi'] = dev_df['hi'].apply(lambda x: unicodedata.normalize('NFKD', x))
dev_df.to_csv("filtered/dev/dev.hi.raw", sep=' ', columns=['hi'], index=False, header=False, encoding='utf-8')

train_df['en'] = train_df['en'].apply(lambda x: unicodedata.normalize('NFKD', x))
train_df.to_csv("filtered/train/train.en.raw", sep=' ', columns=['en'], index=False, header=False, encoding='utf-8')
train_df['hi'] = train_df['hi'].apply(lambda x: unicodedata.normalize('NFKD', x))
train_df.to_csv("filtered/train/train.hi.raw", sep=' ', columns=['hi'], index=False, header=False, encoding='utf-8')

# %%
files = [
    'test/test.en',
    'test/test.hi',
    'dev/dev.en',
    'dev/dev.hi',
    'train/train.en',
    'train/train.hi'
]

# files = ['filtered/test/test.en', 'filtered/test/test.hi']

for file in files:
    data = open(f"filtered/{file}.raw", 'r').readlines()
    new_data = []
    print(f"file={file}.raw")
    for sent in data:
        new_sent = unicodedata.normalize('NFKD', sent.strip()[1:-1])
        # new_sent = sent[1:-1]
        new_data.append(new_sent)
    # open(f"{file}.new", 'w').writelines(new_data)
    print(len(new_data))
    # print(new_data[-1])
    f = open(f"filtered/{file}", 'w')
    write_str = '\n'.join(new_data)
    f.write(write_str)
    f.close()
    # for line in new_data:
    #     f.write(f"{line}\n")


# %%



