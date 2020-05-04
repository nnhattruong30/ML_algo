import numpy as np
import matplotlib.pyplot as plt

# Dinh nghia ham loss
def loss(theta):
    "Ham loss"
    return theta**2 - 4*theta + 10 #(theta - 2)^2 + 6

# Dinh nghia dao ham cua ham loss
def derLoss(theta):
    "Dao ham cua ham loss"
    return 2*theta - 4

# Ve do thi ham loss
theta = np.arange(-12, 16 , 0.1)
plt.plot(theta, loss(theta))

# Khoi tao tham so
theta = -4
alpha = 0.1
eps = 0.001

# Lap
# Cap nhat tham so
# Tim diem dung
while (True):
    theta = theta - alpha*derLoss(theta)
    # Truc quan hoa du lieu
    plt.plot(theta, loss(theta), 'r.')
    plt.pause(0.5)
    if abs(derLoss(theta)) < eps: break

# In ket qua
print('theta toi uu:', theta)
print('Loss toi uu:', loss(theta))
plt.show()