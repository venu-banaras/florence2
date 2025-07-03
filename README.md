# Objective
To learn Florence-2 and infer and train it on own custom dataset of SKU110K

### Explanation
This is Dockerised code to perform inference and training of Florence-2 on any custom dataset. It is however, tested on `SKU110K dataset`. 
1. We initially have YOLO format dataset.
2. We convert it to Florence2 format using the `yolo_to_florence_json.py` script.
3. Then we can simply run `florence2_train.py` to start training Florence-2 on any dataset.
4. Hyperparameters can be tuned in the `florence2_train.py` itself.
5. Using the prompt on which it is finetuned, we can infer Florence-2 on a folder of images and save the outputs in a CSV and also plot them, if needed.


### My experience
Considering this model is a lightweight AGI model, it performed poorly on my test set.
It gave excellent numbers on the validation set of SKU110K, but when I gave it images that were clicked by me, it gave very poor results. However, it is an excellent model, which did give good results even after no augmentations on the dataset.
Training time - 4 hours per epoch on a dataset size of 11983 images, which is fair considering this is a transformer model. 
