import torch.nn.functional as F
import torch
import numpy as np

def JSD_Loss(p, q):
    m = 0.5 * (p + q)
    # compute the JSD Loss
    return 0.5 * (F.kl_div(p.log(), m) + F.kl_div(q.log(), m))


class BagGenerator:
    def __init__(self,device,seed=2032):
        self.device=device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)


    def compute_train_bags(self,n_bags, bag_size,y, train_prevalence=None):
        """Compute bags for training or prediction. If we do not have the class information, compute the bags without taking into account the prevalences"""
        with torch.no_grad():
            if not torch.is_tensor(y):       
                y=torch.IntTensor(y).to(self.device)
            
            #Tensor to return the result. Each bag in a row.
            samples_indexes = torch.zeros((n_bags,bag_size),dtype=torch.int64,device=self.device)
            classes = torch.unique(y)
            n_classes = len(classes)
            
            prevalences = torch.zeros((n_bags,n_classes),device=self.device)
            for i in range(n_bags):
                low = round(bag_size * 0.01)
                high = round(bag_size * 0.99)

                ps = torch.randint(low, high, (n_classes - 1,),generator=self.gen,device=self.device)
                ps = torch.cat([ps, torch.tensor([0, bag_size],device=self.device)])
                ps = torch.sort(ps)[0]
                ps = ps[1:] - ps[:-1] #Number of samples per class
                prevalences[i,:] = ps/bag_size
                already_generated=0
                for n, p in zip(classes, ps):
                    if p != 0:
                        examples_class=torch.where(y==n)[0]
                        samples_indexes[i,already_generated:already_generated+p]=examples_class[torch.randint(len(examples_class),(p,),generator=self.gen,device=self.device)]
                        already_generated+=p
                suffle = torch.randperm(bag_size)
                samples_indexes[i,:]=samples_indexes[i,suffle]

            # Se train_prevalence não for None, ordenar os bags
            if train_prevalence is not None:
            # Calcular a diferença absoluta entre prevalences e train_prevalence
                diff = torch.abs(prevalences - torch.tensor(train_prevalence, device=prevalences.device))

                # Calcular a norma L1 para cada par de prevalências
                norm = torch.sum(diff, dim=1)  # Soma absoluta ao longo das dimensões (L1 norm manual)

                # Ordenar os índices com base nos valores da norma L1
                idxs = torch.argsort(norm, descending=True)

                # Reordenar prevalences e samples_indexes com base nos índices
                prevalences = prevalences[idxs]  # Reordenar prevalences
                samples_indexes = samples_indexes[idxs]  # Reordenar samples_indexes

                
            return samples_indexes,prevalences

    def compute_prediction_bags(self,dataset_size,n_bags,bag_size):
        with torch.no_grad():
            samples_indexes = torch.zeros((n_bags,bag_size),dtype=torch.int64,device=self.device)
            for i in range(n_bags):
                samples_indexes[i,:]=torch.randint(dataset_size,(bag_size,),generator=self.gen,device=self.device)
            return samples_indexes
        

# def compute_val_bags(self, n_bags, bag_size, y, train_prevalence=None):
#     with torch.no_grad():
#         if not torch.is_tensor(y):       
#             y = torch.IntTensor(y).to(self.device)
        
#         # Inicializa os tensores
#         samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
#         classes = torch.unique(y)
#         n_classes = len(classes)
#         prevalences = torch.zeros((n_bags, n_classes), device=self.device)
        
#         for i in range(n_bags):
#             low = round(bag_size * 0.01)
#             high = round(bag_size * 0.99)
            
#             ps = torch.randint(low, high, (n_classes - 1,), generator=self.gen, device=self.device)
#             ps = torch.cat([ps, torch.tensor([0, bag_size], device=self.device)])
#             ps = torch.sort(ps)[0]
#             ps = ps[1:] - ps[:-1]  # Número de amostras por classe
#             prevalences[i, :] = ps / bag_size
#             already_generated = 0
#             for n, p in zip(classes, ps):
#                 if p != 0:
#                     examples_class = torch.where(y == n)[0]
#                     samples_indexes[i, already_generated:already_generated + p] = examples_class[
#                         torch.randint(len(examples_class), (p,), generator=self.gen, device=self.device)]
#                     already_generated += p
#             shuffle = torch.randperm(bag_size)
#             samples_indexes[i, :] = samples_indexes[i, shuffle]

#         # Se train_prevalence não for None, ordenar os bags
#         if train_prevalence is not None:
#         # Calcular a diferença absoluta entre prevalences e train_prevalence
#             diff = torch.abs(prevalences - torch.tensor(train_prevalence, device=prevalences.device))

#             # Calcular a norma L1 para cada par de prevalências
#             norm = torch.sum(diff, dim=1)  # Soma absoluta ao longo das dimensões (L1 norm manual)

#             # Ordenar os índices com base nos valores da norma L1
#             idxs = torch.argsort(norm, descending=True)

#             # Reordenar prevalences e samples_indexes com base nos índices
#             prevalences = prevalences[idxs]  # Reordenar prevalences
#             samples_indexes = samples_indexes[idxs]  # Reordenar samples_indexes


#         return samples_indexes, prevalences
