#!/usr/bin/env python
# coding: utf-8

# In[2]:
#from memory_profiler import profile
import os
import torch
from torch import cuda
import time
from guppy import hpy
import tracemalloc

device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'

memusage = 0
index = 0
mylocs = []


os.environ['SENTENCE_TRANSFORMERS_HOME'] = './.cache'

#time.sleep = lambda *args: print("Here's the sleeping!")
# In[3]:


from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[4]:


MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-03
CHECKPOINT_INTERVAL = 4


# In[5]:


with open('./data/imps.txt', 'r') as f:
    imps = f.read().split('\n')

with open ('./data/decls.txt', 'r') as f:
    decls = f.read().split('\n')
    decls = decls[:500] #since imps is small

#uses the longest tokenized sentence as guideline for padding. return_token_type_ids is used is pair sentence classification
#return_tensors='pt' makes it returns torch.Tensor objects
tokenized_data = tokenizer(imps+decls, padding='max_length', return_token_type_ids=False, return_tensors='pt', max_length=MAX_LEN)


# In[6]:


from datasets import Dataset

labels = torch.tensor([[1, 0] for elm in imps] + [[0, 1] for elm in decls])
#labels = torch.tensor([1 for elm in imps] + [-1 for elm in decls])

dataset = Dataset.from_dict({
    'ids': tokenized_data['input_ids'],
    'masks': tokenized_data['attention_mask'],
    'labels': labels
})

dataset.set_format(type='torch', columns=['ids','masks', 'labels'])

dataset = dataset.train_test_split(train_size=0.8, seed=123)


# In[7]:


dataset['train']


# In[8]:


from transformers import DistilBertModel, DistilBertConfig
DROPOUT = 0.2
ATT_DROPOUT = 0.2
config = DistilBertConfig(dropout=DROPOUT, attention_dropout=ATT_DROPOUT, output_hidden_states=False)
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)


# In[9]:


list(distilbert.modules())


# In[10]:


#freeze current model
for param in distilbert.parameters():
    param.requires_grad = False 


import memory_profiler

def get_current_memory_usage():
    """
    Returns the current memory usage of the program in megabytes (MB).
    """
    mem_usage = memory_profiler.memory_usage(proc=-1, interval=.0005, timeout=.001)
    current_mem_usage = max(mem_usage)  # Get the maximum memory usage value
    global memusage
    memusage = current_mem_usage
    return current_mem_usage

import copy
import cProfile

def object_profiler(myobj):
    attr_holder = []
    for key in vars(myobj):

        new_copy = copy.deepcopy(getattr(myobj, key))
        attr_holder.append(new_copy)

        print(key)
        print(get_current_memory_usage())
    #mylocs.append(hpy().heap())
    #breakpoint()

    


def globals_profiler():
    globals_holder = []
    for elm in globals():

        new_copy = copy.deepcopy(elm)
        globals_holder.append(elm)

        print(elm)
        print(get_current_memory_usage())


# In[11]:


import torch
import torch.nn as nn
import sys
import pprint
import gc

import pstats
#profiler = cProfile.Profile()


# Add a classification head on top of the frozen DistilBertModel
class DistilBertClassifier(torch.nn.Module):
    def __init__(self, distilbert_model):
        super().__init__()
        self.distilbert = distilbert_model
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 2)  #  Binary classification

    #@profile
    def forward(self, input_ids, attention_mask):
        #profiler.enable()
        #mylocs.append(tracemalloc.take_snapshot())
        print("current mem before forward's output: ", get_current_memory_usage())
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        print("current mem after forward's output: ", get_current_memory_usage())
        pooled_output = output.last_hidden_state[:, 0]  # Grab only the first token of Distilbert's output, which is the [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        print("current mem before fiest object_profiler call: ", get_current_memory_usage())
        #object_profiler(self.distilbert)
        #mylocs.append(hpy().heap())
        #import ipdb; ipdb.set_trace()
        #globals_profiler()
        #breakpoint()

        print("current mem after first object_profiler call, before second: ", get_current_memory_usage())
        #object_profiler(self.distilbert)
        print("current mem after second object_profiler call: ", get_current_memory_usage())


        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats('tottime')
        #stats.print_stats()
        #mylocs.append(tracemalloc.take_snapshot())
        breakpoint()
        return logits

# Create an instance of the DistilBertClassifier
classifier_model = DistilBertClassifier(distilbert)


# In[12]:

def find_available_filename(name, directory_path):

   i = 1
   while True:
       filename = os.path.join(directory_path, f"{name}_{i}.pt")
       if not os.path.exists(filename):
           return filename
       i += 1

#@profile
def save_checkpoint(epoch, checkpoint_interval, loss, model, optimizer):
    if epoch % checkpoint_interval == 0 and epoch != 0:
        filename = find_available_filename(f"distilbert_checkpoint", "models/distilbert-base-uncased/checkpoints")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, filename)


# In[13]:


from tqdm import tqdm


def loss_fn(outputs, targets):
    # print("---loss_fn inputs---")
    # print("outputs: ", outputs)
    # print("targets: ", targets)
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  classifier_model.parameters(), lr=LEARNING_RATE)




#@profile
def train(epoch, training_loader, model, optimizer):
    model.train()
    loss = None
    for _,data in tqdm(enumerate(training_loader, 0)):

        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['masks'].to(device, dtype = torch.long)
        targets = data['labels'].to(device, dtype = torch.float)

        print("current mem before trains's output: ", get_current_memory_usage())
        outputs = model(ids, mask)
        print("current mem after trains's output: ", get_current_memory_usage())


        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()
    save_checkpoint(epoch, CHECKPOINT_INTERVAL, loss, model, optimizer)

def validation(testing_loader, model):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['masks'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


# In[14]:


from torch.utils.data import DataLoader, RandomSampler


train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'num_workers': 0
}

train_sampler = RandomSampler(dataset['train'], replacement=False)
test_sampler = RandomSampler(dataset['test'], replacement=False)

training_loader = DataLoader(dataset['train'], sampler=train_sampler, **train_params)
testing_loader = DataLoader(dataset['test'], sampler=test_sampler, **train_params)


# In[15]:



# In[17]:


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
modules = classifier_model.distilbert.modules()
list(modules)[0].transformer.layer[5].output_layer_norm.register_forward_hook(get_activation('output_layer_norm'))


# In[18]:


activation


# In[19]:


print(device)


# In[20]:


import cProfile
import sys


def main():
 print("DEVICE: ", device)
 for epoch in range(EPOCHS):
     train(epoch, training_loader, classifier_model, optimizer)

# sys.stdout = open("profiling_output.txt", "w")
# cProfile.run('main()', sort='cumtime')
# sys.stdout = sys.__stdout__

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

print("testiiinngg")

tracemalloc.start()

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        main()
        prof_res = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)

        print("printing prof_res")
        print(prof_res)

with open("./profiler_results.txt", "w") as file:
    file.write(prof_res)


# In[ ]:


# from sklearn import metrics


# val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
# val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))

# print(f"Hamming Score = {val_hamming_score}")

