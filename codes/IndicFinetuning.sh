mkdir -p /scratch/monilg
cd /scratch/monilg

rm -r finetuning2

mkdir finetuning2
cd finetuning2

# clone the repo for running finetuning
git clone https://github.com/AI4Bharat/indicTrans.git
cd indicTrans
# clone requirements repositories
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
git clone https://github.com/rsennrich/subword-nmt.git
cd ..

# ========================
#apt install tree

# Install the necessary libraries
#pip install sacremoses pandas mock sacrebleu tensorboardX pyarrow indic-nlp-library
# Install fairseq from source

git clone https://github.com/pytorch/fairseq.git
cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
pip install --editable ./
cd ..

# =====================
# if you want to finetune indic-en models, use the link below
wget https://storage.googleapis.com/samanantar-public/V0.2/models/en-indic.zip
unzip -qq en-indic.zip

# ======================

tar -xzvf ~/iitb_parallel_normal.tar.gz
# ==========================================

cp train/train_nopunc.en train/train.en
cp train/train_nopunc.hi train/train.hi

cp dev/dev_nopunc.en dev/dev.en
cp dev/dev_nopunc.hi dev/dev.hi

cp test/test_nopunc.en test/test.en
cp test/test_nopunc.hi test/test.hi

mkdir -p dataset
mv train dataset
mv dev dataset
mv test dataset

# lets cd to indicTrans
cd indicTrans
# ============
#cd ../dataset/
#head train/train.en > temp
#mv temp train/train.en

#head train/train.hi >temp
#mv temp train/train.hi

#head dev/dev.en >temp
#mv temp dev/dev.en
#head dev/dev.hi >temp
#mv temp dev/dev.hi

#head test/test.hi >temp
#mv temp test/test.hi
#head test/test.en >temp
#mv temp test/test.en

#cd ../indicTrans
# ==============================
# all the data preparation happens in this cell
exp_dir=../dataset
src_lang=en
tgt_lang=indic

# change this to indic-en, if you have downloaded the indic-en dir or m2m if you have downloaded the indic2indic model
download_dir=../en-indic

train_data_dir=$exp_dir/train
dev_data_dir=$exp_dir/dev
test_data_dir=$exp_dir/test


echo "Running experiment ${exp_dir} on ${src_lang} to ${tgt_lang}"


train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

out_data_dir=$/exp_dir/final_bin

mkdir -p $train_processed_dir
mkdir -p $devtest_processed_dir
mkdir -p $out_data_dir

# indic languages.
langs=(hi)

for lang in ${langs[@]};do
	if [ $src_lang == en ]; then
		tgt_lang=$lang
	else
		src_lang=$lang
	fi

	train_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	devtest_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	mkdir -p $train_norm_dir
	mkdir -p $devtest_norm_dir


    # preprocessing pretokenizes the input (we use moses tokenizer for en and indicnlp lib for indic languages)
    # after pretokenization, we use indicnlp to transliterate all the indic data to devnagiri script

	# train preprocessing
	train_infname_src=$train_data_dir/train.$src_lang
	train_infname_tgt=$train_data_dir/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang
	echo "Applying normalization and script conversion for train $lang"
	input_size=`python scripts/preprocess_translate.py $train_infname_src $train_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $train_infname_tgt $train_outfname_tgt $tgt_lang true`
	echo "Number of sentences in train $lang: $input_size"

	# dev preprocessing
	dev_infname_src=$dev_data_dir/dev.$src_lang
	dev_infname_tgt=$dev_data_dir/dev.$tgt_lang
	dev_outfname_src=$devtest_norm_dir/dev.$src_lang
	dev_outfname_tgt=$devtest_norm_dir/dev.$tgt_lang
	echo "Applying normalization and script conversion for dev $lang"
	input_size=`python scripts/preprocess_translate.py $dev_infname_src $dev_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $dev_infname_tgt $dev_outfname_tgt $tgt_lang true`
	echo "Number of sentences in dev $lang: $input_size"

	# test preprocessing
	test_infname_src=$test_data_dir/test.$src_lang
	test_infname_tgt=$test_data_dir/test.$tgt_lang
	test_outfname_src=$devtest_norm_dir/test.$src_lang
	test_outfname_tgt=$devtest_norm_dir/test.$tgt_lang
	echo "Applying normalization and script conversion for test $lang"
	input_size=`python scripts/preprocess_translate.py $test_infname_src $test_outfname_src $src_lang true`
	input_size=`python scripts/preprocess_translate.py $test_infname_tgt $test_outfname_tgt $tgt_lang true`
	echo "Number of sentences in test $lang: $input_size"
done




# Now that we have preprocessed all the data, we can now merge these different text files into one
# ie. for en-as, we have train.en and corresponding train.as, similarly for en-bn, we have train.en and corresponding train.bn
# now we will concatenate all this into en-X where train.SRC will have all the en (src) training data and train.TGT will have all the concatenated indic lang data

python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'train'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'dev'
python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang 'test'

# use the vocab from downloaded dir
cp -r $download_dir/vocab $exp_dir


echo "Applying bpe to the new finetuning data"
bash apply_single_bpe_traindevtest_notag.sh $exp_dir

mkdir -p $exp_dir/final

# We also add special tags to indicate the source and target language in the inputs
#  Eg: to translate a sentence from english to hindi , the input would be   __src__en__   __tgt__hi__ <en bpe tokens>

echo "Adding language tags"
python scripts/add_joint_tags_translate.py $exp_dir 'train'
python scripts/add_joint_tags_translate.py $exp_dir 'dev'
python scripts/add_joint_tags_translate.py $exp_dir 'test'



data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

rm -rf $out_data_dir

# binarizing the new data (train, dev and test) using dictionary from the download dir

 num_workers=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`

data_dir=$exp_dir/final
out_data_dir=$exp_dir/final_bin

# rm -rf $out_data_dir

echo "Binarizing data. This will take some time depending on the size of finetuning data"
fairseq-preprocess --source-lang SRC --target-lang TGT \
 --trainpref $data_dir/train --validpref $data_dir/dev --testpref $data_dir/test \
 --destdir $out_data_dir --workers $num_workers 

# --srcdict $download_dir/final_bin/dict.SRC.txt --tgtdict $download_dir/final_bin/dict.TGT.txt --thresholdtgt 5 --thresholdsrc 5  


# ====================================
# Finetuning the model

fairseq-train  ../dataset/final_bin  --arch transformer  --keep-last-epochs 5 --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --save-interval=50  --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1  --save-dir ../dataset/model --skip-invalid-size-inputs-valid-test --update-freq=2 --warmup-init-lr 1e-07 --max-tokens 2048 --fp16

# =======================================================================
# To test the models after training, you can use joint_translate.sh



# joint_translate takes src_file, output_fname, src_lang, tgt_lang, model_folder as inputs
# src_file -> input text file to be translated
# output_fname -> name of the output file (will get created) containing the model predictions
# src_lang -> source lang code of the input text ( in this case we are using en-indic model and hence src_lang would be 'en')
# tgt_lang -> target lang code of the input text ( tgt lang for en-indic model would be any of the 11 indic langs we trained on:
#              as, bn, hi, gu, kn, ml, mr, or, pa, ta, te)
# supported languages are:
#              as - assamese, bn - bengali, gu - gujarathi, hi - hindi, kn - kannada, 
#              ml - malayalam, mr - marathi, or - oriya, pa - punjabi, ta - tamil, te - telugu

# model_dir -> the directory containing the model and the vocab files

# Note: if the translation is taking a lot of time, please tune the buffer_size and batch_size parameter for fairseq-interactive defined inside this joint_translate script


# here we are translating the english sentences to hindi
bash joint_translate.sh $exp_dir/test/test.en en_hi_outputs.txt 'en' 'hi' $exp_dir

# ===================================================================


# to compute bleu scores for the predicitions with a reference file, use the following command
# arguments:
# pred_fname: file that contains model predictions
# ref_fname: file that contains references
# src_lang and tgt_lang : the source and target language

bash compute_bleu.sh en_hi_outputs.txt $exp_dir/test/test.hi 'en' 'hi'
