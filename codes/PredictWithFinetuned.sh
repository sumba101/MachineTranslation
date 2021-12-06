mkdir -p /scratch/monilg
cd /scratch/monilg

rm -rf indicTrans finetunedata
scp monilg@ada.iiit.ac.in:/share1/monilg/finetuning_Rich.zip . 
# The above is with rich text transformer
# rm -rf finetuning_Rich
unzip -qq finetuning_Rich.zip


git clone https://github.com/pytorch/fairseq.git
cd fairseq
!git checkout da9eaba12d82b9bfc1442f0e2c6fc1b895f4d35d
pip install --editable ./
cd ..




cd indicTrans

# ==============================
# all the data preparation happens in this cell
exp_dir=../finetunedata
src_lang=en
tgt_lang=indic

train_data_dir=$exp_dir/train
dev_data_dir=$exp_dir/dev
test_data_dir=$exp_dir/test

train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

out_data_dir=$exp_dir/final_bin

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
bash joint_translate.sh ~/SampleFiles/sample_1.en prediction_finetuned_1.hi 'en' 'hi' $exp_dir
bash joint_translate.sh ~/SampleFiles/sample_2.en prediction_finetuned_2.hi 'en' 'hi' $exp_dir
bash joint_translate.sh ~/SampleFiles/sample_3.en prediction_finetuned_3.hi 'en' 'hi' $exp_dir
# bash joint_translate.sh ~/SampleFiles/sample_4.en prediction_finetuned_4.hi 'en' 'hi' $exp_dir
# bash joint_translate.sh ~/SampleFiles/sample_5.en prediction_finetuned_5.hi 'en' 'hi' $exp_dir

cp prediction_* ~/SampleFiles/fintuning_op/

