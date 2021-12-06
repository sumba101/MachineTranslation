# To run, place this in the directory containing the extracted .en and .hi files from IITB parallel dataset
# Alternatively, change IPEN and IPHI to point to the English and Hindi files respectively
# Note: Splits are randomly generated, running the same code twice will not produce identical splits. 



# %%
import pandas as pd
import string, re, unicodedata, os

IPEN = "IITB.en-hi.en"
IPHI = "IITB.en-hi.hi"

pref = os.getcwd()

for dir in ['train1', 'dev1', 'test1']:
    try:
        os.makedirs(os.path.join(pref, dir), exist_ok = True)
    except OSError as error:
        print(f"Directory creation failed because of following error \n{error}")

exit()

print("hi")
# %%
en_sent = open(IPEN, 'r').readlines()
hi_sent = open(IPHI, 'r').readlines()


# %%
full_df = pd.DataFrame(
    {
        'en':en_sent,
        'hi':hi_sent
    }
)
print(f"total length= {len(full_df)}")

full_df['en'].astype(str)
filt_df = full_df[full_df['en'].str.split().str.len() > 7]
print(f"length after filetering for 7 words: {len(filt_df)}")
# %%
dev_test_df = filt_df.sample(frac = 0.4)


# %%
train_df = filt_df.drop(dev_test_df.index)
print(str(len(train_df)) + "rows in train")


# %%
dev_df = dev_test_df.sample(frac = (0.5))
print(str(len(dev_df)) + "rows in dev")


# %%
test_df = dev_test_df.drop(dev_df.index)
print(str(len(test_df)) + "rows in test")


# %%
train_en = open('train/train_nopunc.en','w')
train_hi = open('train/train_nopunc.hi', 'w')
dev_en = open('dev/dev_nopunc.en', 'w')
dev_hi = open('dev/dev_nopunc.hi', 'w')
test_en = open('test/test_nopunc.en', 'w')
test_hi = open('test/test_nopunc.hi', 'w')    


train_en_raw = open('train/train_raw.en','w')
train_hi_raw = open('train/train_raw.hi', 'w')
dev_en_raw = open('dev/dev_raw.en', 'w')
dev_hi_raw = open('dev/dev_raw.hi', 'w')
test_en_raw = open('test/test_raw.en', 'w')
test_hi_raw = open('test/test_raw.hi', 'w')     


# %%
extras = ['|', '।', '–', '’', '‘']
punc_list = list(string.punctuation)
punc_list = punc_list + extras
puncts = re.compile(f"[{''.join(punc_list)}]")

def rem_puncts(inp):
    return re.sub(puncts, '', inp).strip()
# %%
for index, row in test_df.iterrows():
    test_en_raw.write(f"{unicodedata.normalize('NFKD', row['en'].strip().lower())}\n")
    test_hi_raw.write(f"{unicodedata.normalize('NFKD', row['hi'].strip().lower())}\n")
    test_en.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['hi'].strip().lower()))}\n")
    test_hi.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['hi'].strip().lower()))}\n")
print("test split complete")

# %%
for index, row in dev_df.iterrows():
    dev_en_raw.write(f"{unicodedata.normalize('NFKD', row['en'].strip())}\n")
    dev_hi_raw.write(f"{unicodedata.normalize('NFKD', row['hi'].strip())}\n")
    dev_en.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['en'].strip().lower()))}\n")
    dev_hi.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['hi'].strip().lower()))}\n")
print("dev split complete")

i = 0
for index, row in train_df.iterrows():
    i += 1
    if i%50000 == 0:
        print(f"{i} sentences completed in train split")
    train_en_raw.write(f"{unicodedata.normalize('NFKD', row['en'].strip().lower())}\n")
    train_hi_raw.write(f"{unicodedata.normalize('NFKD', row['hi'].strip().lower())}\n")
    train_en.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['en'].strip().lower()))}\n")
    train_hi.write(f"{rem_puncts(unicodedata.normalize('NFKD', row['hi'].strip().lower()))}\n")
print("train split complete")
# %%
train_en.close()
train_hi.close()
dev_en.close()
dev_hi.close()
test_en.close()
test_hi.close()
train_en_raw.close()
train_hi_raw.close()
dev_en_raw.close()
dev_hi_raw.close()
test_en_raw.close()
test_hi_raw.close() 


# %%



