# Visual assignment 1

I found a collection of memes from kaggle datasets, which i split, as shown in the notebook. 
I did my best to make a .py file as well, turning the splitting into a function. I am not sure it will run in terminal, but i'm hoping Ross will clarify and i'll adapt it

# Visual assignment 2
Create the virtual environment and include the flower images in the same folder. The .py file also gives you the opportunity to load the function get_img_histsim into you pyhton environment

# Visual assignment 4
```
git clone https://github.com/Guscode/Visual_assignments.git
cd Visual_assignments/assignmnet_4
```

Then create and open the virtual environment lang_ass
```
bash ./create_clf_env.sh 
source ./clf_env/bin/activate 
```

In the virtual environment you can run lr_mnist.py or nn_mnist.py from the src folder specifying path to mnist data.
I have provided mnist_mini.csv for testing it without running for 5 hours. 
in lr-mnist.py you're able to specify:
- --mnist -m: path to mnist dataset
- --output -o: path to output folder
- --solver -s: solver algorithm 
- --test_split -ts: test split in decimal
- --penalty -p: penalty norm
- --test_image -t: path to test image 
```
python src/lr-mnist.py -m mnist_mini.csv -ts 0.1 -t test.png -o outputs
```


in nn-mnist you're able to specify:
- --mnist -m: path to mnist dataset
- --output -o: path to output folder
- --test_split -ts: test split in decimal
- --layers -l: hidden layers
- --epochs -e: amount of epochs
- --test_image -t: path to test image 
- --save_model_path -s: path to store saved model
```
python src/nn-mnist.py -m mnist_mini.csv -ts 0.1 -t test.png -s outputs
```

# Visual Assignment 5
I created a .py script which can be run using the cv101 virtual environment on worker02.
You can specify:
- --training_path -t: path to training folder
- --validation_path -v: path to validation folder
- --epochs -e: number of epochs 
