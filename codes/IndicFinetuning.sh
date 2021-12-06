mkdir -p /scratch/monilg
cd /scratch/monilg

scp monilg@ada.iiit.ac.in:/share1/monilg/finetuning.zip .

unzip -qq finetuning.zip

cd finetuning

#git clone https://github.com/pytorch/fairseq.git
cd fairseq
# !git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
pip install --editable ./
cd ..

mkdir finetunedata

#unzip -qq en-indic.zip

# ======================

tar -xzvf ~/en_hi_RT_split_fixed.tar.gz
# ==========================================
#Remove old dataset
rm -r dataset/train
rm -r dataset/dev
rm -r dataset/test

mv train finetunedata
mv dev finetunedata
mv test finetunedata

# lets cd to indicTrans
cd indicTrans

# ==============================
# all the data preparation happens in this cell
exp_dir=../finetunedata
src_lang=en
tgt_lang=indic

# change this to indic-en, if you have downloaded the indic-en dir or m2m if you have downloaded the indic2indic model
download_dir=../dataset

train_data_dir=$exp_dir/train
dev_data_dir=$exp_dir/dev
test_data_dir=$exp_dir/test


echo "Running experiment ${exp_dir} on ${src_lang} to ${tgt_lang}"


train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

out_data_dir=$exp_dir/final_bin

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

# remove the vocab from downloaded dir
cp -r $download_dir/vocab $exp_dir
#rm -r vocab

echo "Applying bpe to the new finetuning data"

# Changing the script
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


echo "Binarizing data. This will take some time depending on the size of finetuning data"
fairseq-preprocess --source-lang SRC --target-lang TGT \
 --trainpref $data_dir/train --validpref $data_dir/dev --testpref $data_dir/test \
 --destdir $out_data_dir --workers $num_workers \
 --srcdict $download_dir/final_bin/dict.SRC.txt --tgtdict $download_dir/final_bin/dict.TGT.txt --thresholdtgt 5 --thresholdsrc 5  

# ====================================
# Finetuning the model


fairseq-train $out_data_dir \
--max-update=5 \
--save-interval=1 \
--share-decoder-input-output-embed \
--arch=transformer \
--criterion=label_smoothed_cross_entropy \
--lr-scheduler=inverse_sqrt \
--label-smoothing=0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 3e-5 \
--warmup-updates 4000 \
--dropout 0.2 \
--save-dir $exp_dir/model \
--skip-invalid-size-inputs-valid-test \
--fp16 \
--update-freq=1 \
--distributed-world-size 4 \
--max-tokens 2048 \
--restore-file ../dataset/model/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer

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

cp en_hi_outputs.txt ~/finetuned_translated_output.txt
cp $exp_dir/test/test.en ~/finetuned_source_english.txt
cp $exp_dir/test/test.hi ~/finetuned_target_hindi.txt


