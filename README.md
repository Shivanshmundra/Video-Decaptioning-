# Chalearn LAP Inpainting Competition Track 2 - Video decaptioning

This is repository for Track2 of Chalearn LAP Inpainting Competition - Video decaptioning

# Instruction for Training

  - Firstly, convert all your video files (video with subtitles and without subtitles) into **tensorflow records** in folder named `train_records` in source folder by running `preprocess_train_images.py`.
  - Once tf records are made, you can start training by running `main.py` 
  - You can also specify attributes according to hardware i.e. `python main.py --use_tfrecords True --batch_size 16`
  - You can edit model architecture in `model.py`
  
# Instruction for Testing

  - Once model is trained and weight files are updated you can call `test.py` to see outcome of trained model.
  - Specify source destination of test videos(with subtitles) in `model.py` (currently it is `~/Inpaintin/dev/X`)
  - Output videos will be stored in folder named `out_video` with name changed as `'X'` will be replaced by `'Y'` in name.

