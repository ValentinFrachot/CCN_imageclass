import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),#Convertion d'image en tenseur PyTorch
    transforms.Normalize((0.5,), (0.5,))# Normalisation de chaque canal (R, G, B) avec une moyenne et un écart-type
])
#chargement des données d'entrainement
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)# Chargement des données d'entraînement avec mini-lots
#chargement des données de test

testset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False) # Chargement des données de test sans mélange

class MonCNN(nn.Module):
    def __init__(self):
        super(MonCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(64,64,kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc2 = nn.Linear(3136, 10)
    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x


net = MonCNN() #appel du modèle
criterion = nn.CrossEntropyLoss() #fonction perte pour l'entrainement
#optimiseur Adam, ajuste les poids du modèle
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Optimiseur Adam avec un taux d'apprentissage de 0.001

#boucle d'entrainement
train_losses = []  # Pour tracer la courbe de perte

for epoch in range(5):  # 5 passes sur le dataset
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    train_losses.append(epoch_loss / len(trainloader))  # Moyenne sur toute l’époque

correct = 0
total = 0
class_correct = [0] * 10  # Liste pour stocker les prédictions correctes par classe
class_total = [0] * 10    # Liste pour stocker le total d'images par classe

# Évaluation de la performance du modèle
with torch.no_grad():  # Pas besoin de calculer les gradients lors de l'évaluation
    for data in testloader:  # Boucle sur les données de test
        images, labels = data  # Récupère les images et leurs labels
        outputs = net(images)  # Passe les images dans le modèle
        _, predicted = torch.max(outputs, 1)  # Prédiction de la classe la plus probable
        total += labels.size(0)  # Ajoute le nombre d'images traitées
        correct += (predicted == labels).sum().item()  # Compte les bonnes prédictions

        # Met à jour les tableaux des classes
        for i in range(len(labels)):
            label = labels[i]
            class_total[label] += 1  # Incrémente le nombre total d'images pour cette classe
            if predicted[i] == label:
                class_correct[label] += 1  # Incrémente le nombre d'images correctement classées pour cette classe

# Affichage de l'accuracy globale
print(f'Accuracy: {100 * correct / total:.2f}%')

# Affichage de l'accuracy par classe
for i in range(10):
    if class_total[i] > 0:
        print(f'Accuracy for class {i} ({classes[i]}): {100 * class_correct[i] / class_total[i]:.2f}%')

plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Évolution de la perte pendant l'entraînement")
plt.xlabel("Époque")
plt.ylabel("Perte moyenne")
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy par classe en graphique
accuracy_per_class = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]

plt.figure(figsize=(10, 5))
plt.bar(classes, accuracy_per_class)
plt.title("Accuracy par classe")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
