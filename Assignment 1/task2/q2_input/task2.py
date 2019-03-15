import tensorflow as tf
import numpy as np

m = np.load("masses.npy")
p = np.load("positions.npy")
v = np.load("velocities.npy")

masses = tf.Variable(m, dtype = tf.float32) 
positions = tf.Variable(p, dtype = tf.float32) 
velocities = tf.Variable(v, dtype = tf.float32)

e1_cap_positions = positions[:, 0]
e2_cap_positions = positions[:, 1] 
e1_cap_velocities = velocities[:, 0]
e2_cap_velocities = velocities[:, 1] 

e1_dash_cap_positions = tf.transpose([e1_cap_positions]) - e1_cap_positions
e2_dash_cap_positions = tf.transpose([e2_cap_positions]) - e2_cap_positions

Identity = tf.eye(100,100,dtype=tf.float32)  
Threshold_distance = 0.1
delta_time = 10 ** (-4)
G = tf.Variable(6.67 * (10 ** 5))

distances = tf.sqrt(tf.add(tf.square(e1_dash_cap_positions),tf.square(e2_dash_cap_positions))) 
distances = tf.add(distances,Identity)

flag = tf.Variable(0) 
System_check = tf.cond(tf.size(tf.where(tf.less_equal(distances, Threshold_distance))) > 0, lambda: tf.assign(flag, 1), lambda: tf.assign(flag, 0))

distances = tf.subtract(tf.pow(tf.reciprocal(distances), 3),Identity)

e1_cap_acceleration = tf.matmul(G*tf.multiply(distances, e1_dash_cap_positions), masses)
e2_cap_acceleration = tf.matmul(G*tf.multiply(distances, e2_dash_cap_positions), masses)
acceleration = tf.concat([tf.reshape(e1_cap_acceleration, [100, 1]),tf.reshape(e2_cap_acceleration, [100, 1])] , 1) 

new_positions = tf.assign(positions, positions + delta_time * velocities + (1 / 2) * (delta_time * delta_time) * acceleration)
new_velocities = tf.assign(velocities, velocities + delta_time * acceleration)

sess = tf.Session() 
sess.run(tf.global_variables_initializer()) 
threshold_cond = 0 
counter = 0

while (threshold_cond == 0):
  threshold_cond = sess.run(System_check)
  finalposition, finalvelocity = sess.run([new_positions, new_velocities]) # new values of r and v
  counter += 1
  
np.save("finalposition.npy",finalposition)
np.save("finalvelocity.npy",finalvelocity)
