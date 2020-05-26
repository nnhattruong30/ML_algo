import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Buoc 0: Phat sinh du lieu va truc quan hoa
S1= np.array([[10],[-2]])
S2= np.array([[2],[8]])
S3= np.array([[12],[8]])
S4= np.array([[-2],[0]])

n_sample = 10

pts1 = S1 + np.random.normal(0, 1.5, size=(2,n_sample))
pts2 = S2 + np.random.normal(0, 1.5, size=(2,n_sample))
pts3 = S3 + np.random.normal(0, 1.5, size=(2,n_sample))
pts4 = S4 + np.random.normal(0, 1.5, size=(2,n_sample))

K = 4 # So luong phan lop
# Tao ma tran dac trung dau vao X va gia tri phan loai dau ra Y
X = np.concatenate(([np.ones(4*n_sample)], np.concatenate((pts1, pts2, pts3, pts4), axis=1)))
Y = np.concatenate((np.zeros(n_sample), np.ones(n_sample), 2*np.ones(n_sample), 3*np.ones(n_sample)))

# Chuyen doi gia tri phan lop sang dang one-hot vector
# Vi du: chuyen cac gia tri phan lop [1 2 0] sang dang
#              [[0 0 1],
#               [1 0 0],
#               [0 1 0]]
#               [0 0 0]]
Y = Y.astype(int)
Y_oh = np.zeros((Y.max()+1, Y.size))
Y_oh[Y, np.arange(Y.size)] = 1

# Truc quan hoa cac du lieu mau
plt.axis('equal')
plt.xlim((-5, 15))
plt.ylim((-5, 15))
plt.plot(pts1[0,:], pts1[1,:], 'ro')
plt.plot(pts2[0,:], pts2[1,:], 'b+')
plt.plot(pts3[0,:], pts3[1,:], 'g^')
plt.plot(pts4[0,:], pts4[1,:], 'mx')

@tf.function
def Loss(theta):
    # CODE HERE: Tinh gia tri du doan Y~ = softmax(theta' * X)
    # Goi y: 1 dong code, su dung cac ham tf.nn.softmax, tf.transpose, tf.linalg.matmul
    # Y_ = ....
    

    # CODE HERE: Tinh do loi dua tren gia tri du doan Y~ va gia tri thuc te duoi dạng one-hot vector Y_oh
    # loss = - average(Y_oh x log(Y_))
    # Goi y: 1 dong code, su dung cac ham tf.math.log, tf.reduce_mean
    # loss = ...
    
    return loss


# Buoc 1: Khoi tao tham so theta
# CODE HERE: khoi tao ma tran ngau nhien kich thuoc 3 x 4
# Goi y: su dung 1 dong code
# theta = ...


def visualize_model(theta):
    theta_ = theta.numpy()
    x1_vis = np.array([-3, 13.0])
    colors = ['r-', 'b-', 'g-', 'm-']
    # Draw lines
    for line in range(K):
        x2_vis = -theta_[1][line]/theta_[2][line]*x1_vis - theta_[0][line]/theta_[2][line]
        plt.plot(x1_vis, x2_vis, colors[line])
    plt.plot(pts1[0,:], pts1[1,:], 'ro')
    plt.plot(pts2[0,:], pts2[1,:], 'b+')
    plt.plot(pts3[0,:], pts3[1,:], 'g^')
    plt.plot(pts4[0,:], pts4[1,:], 'mx')

visualize_model(theta)
# Buoc 2: Lap
# Buoc 2.1: Tinh Gradient
# Buoc 2.2: Dung gradient cap nhat tham so
# Buoc 2.3: Dung gradient kiem tra diem dung
eps = 0.000001
opt = tf.keras.optimizers.SGD(learning_rate = 0.01)

nloop = 0
while True:
    # Khoi tao Gradient Tape
    with tf.GradientTape() as tape:
        # Tinh loss
        loss = Loss(theta)
        
	# CODE HERE: Tinh Gradient bang Gradient Tape
    # Goi y: 1 dong code
    # grads = ...
    

    # CODE HERE: Cap nhat tham so cua mo hinh su dung vector gradient
    # Goi y: 1 dong code, su dung ham apply_gradient cua doi tuong 'opt'
    
    
    # Truc quan hoa mo hinh trong qua trinh train
    nloop = nloop + 1
    if nloop == 1 or nloop % 100 == 0:
        visualize_model(theta)
        plt.pause(0.3)
    
    # Xac dinh diem dung
    if abs(grads[0].numpy()[0][0]) < eps and abs(grads[0].numpy()[1][0]) < eps and abs(grads[0].numpy()[2][0]) < eps and abs(grads[0].numpy()[0][1]) < eps and abs(grads[0].numpy()[1][1]) < eps and abs(grads[0].numpy()[2][1]) < eps:
        break

print('Theta*: ', theta.numpy())
visualize_model(theta)
plt.pause(0.3)
plt.show()
plt.show()

def predict(x, theta):
    # CODE HERE: Cai dat ham du doan phan loai cua mot mau du lieu dau vao voi tham so theta da duoc huan luyen
    # Goi y: 1 dong code, su dung np.matmul, np.transpose
    #y_ = ...
    
    return y_

# Test thu mo hinh
x_test = np.array([[1.0],[-5.0], [-5.0]])
res = predict(x_test, theta)
idx = np.argmax(res)

print('Nhan phan loai cua feature ', x_test, 'la: ', idx)
