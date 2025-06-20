# python 04_batch_renamer_test.py --folder_in ../../pdfs/BondForms -k 1 --dry --skip-existing 
# python 04_batch_renamer_test.py --folder_in ../../pdfs/BondForms -k 1 --skip-existing  --gpu --similarity euclidean

python ./scripts/04_batch_renamer_test_ocr.py --folder_in ../../pdfs/BondFormsSpecifics -k 1 --gpu --threshold 0.6   --similarity cosine