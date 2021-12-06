mkdir -p /scratch/monilg
cd /scratch/monilg

rm -rf finetuning2/
scp monilg@ada.iiit.ac.in:/share1/monilg/finetuning2.zip . 
unzip -qq finetuning2.zip


git clone https://github.com/pytorch/fairseq.git
cd fairseq
!git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
pip install --editable ./
cd ..



cd finetuning2/indicTrans
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

bash joint_translate.sh ~/SampleFiles/sample_1.en prediction_raw_1.hi 'en' 'hi' $exp_dir
bash joint_translate.sh ~/SampleFiles/sample_2.en prediction_raw_2.hi 'en' 'hi' $exp_dir
bash joint_translate.sh ~/SampleFiles/sample_3.en prediction_raw_3.hi 'en' 'hi' $exp_dir
# bash joint_translate.sh ~/SampleFiles/sample_4.en prediction_raw_4.hi 'en' 'hi' $exp_dir
# bash joint_translate.sh ~/SampleFiles/sample_5.en prediction_raw_5.hi 'en' 'hi' $exp_dir

cp prediction_* ~/SampleFiles/nopunc_op/

