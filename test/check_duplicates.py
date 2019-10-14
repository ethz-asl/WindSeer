import numpy as np

train_name = 'train.txt'
validation_name = 'validation.txt'
test_name = 'test.txt'

def main():
    with open(train_name) as f:
        raw = f.readlines()
        train_samples = []
        for name in raw:
            train_samples.append("_".join(name.split("_", 3)[:3]))

    with open(validation_name) as f:
        raw = f.readlines()
        validation_samples = []
        for name in raw:
            validation_samples.append("_".join(name.split("_", 3)[:3]))

    with open(test_name) as f:
        raw = f.readlines()
        test_samples = []
        for name in raw:
            test_samples.append("_".join(name.split("_", 3)[:3]))

    print([i for i in train_samples if i in validation_samples])
    print([i for i in test_samples if i in validation_samples])
    print([i for i in test_samples if i in train_samples])

    test_samples.append(train_samples[-1])
    print([i for i in test_samples if i in train_samples])
    
    import pdb
    pdb.set_trace()

if __name__== "__main__":
  main()