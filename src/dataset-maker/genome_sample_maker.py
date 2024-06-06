import random
#random.seed(0)

def genome_sammple_maker():
    chars = ('A','T','G','C')
    def randchars(chars, length):
        return ''.join(random.choices(chars, k=length))
    sample_num = 96
    X = [randchars(chars, 15) for i in range(sample_num)]
    y = [random.randint(16, 47) for i in range(sample_num)]
    return X,y
