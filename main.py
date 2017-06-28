# -*- coding: utf-8 -*-
import numpy as np
import nnabla as nn

import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from input_data import InputData

# データサイズ系の定数
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
LABEL_NUM = 10
BATCH_SIZE = 64
# 訓練回数
NUM_STEP = 10000
# ハイパーパラメータ
DECAY_RATE = 1e-5

def main():
  # 入力データのshape定義
  x = nn.variable.Variable([BATCH_SIZE, IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_HEIGHT])
  # ラベルのshape定義
  t = nn.variable.Variable([BATCH_SIZE, LABEL_NUM])

  pred = convolution(x)
  loss_ = loss(pred, t)

  solver = S.Adam()
  solver.set_parameters(nn.get_parameters())

  data = InputData()

  for i in range(NUM_STEP):
    # 100STEP毎にテスト実施
    if i % 100 == 0:
      l = 0
      a = 0
      for k, (t.d, x.d) in enumerate(data.test_data()):
        loss_.forward()
        l += loss_.d
        a += accuracy(pred, t)
      print("Step: %05d Test loss: %0.05f Test accuracy: %0.05f" % (i, l / k, a / k))
    t.d, x.d = data.next_batch()
    loss_.forward()
    solver.zero_grad()
    loss_.backward()
    solver.weight_decay(DECAY_RATE)
    solver.update()
    if i % 10 == 0:
      print("Step: %05d Train loss: %0.05f Train accuracy: %0.05f" % (i, loss_.d, accuracy(pred, t)))
    


def convolution(x):
  x = x.reshape([BATCH_SIZE, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])
  with nn.parameter_scope("conv1"):
    output = PF.convolution(x, 16, (5, 5), stride=(2, 2), pad=(1, 1))
    output = F.relu(output)
  
  with nn.parameter_scope("conv2"):
    output = PF.convolution(output, 32, (3, 3), stride=(1, 1), pad=(1, 1))
    output = F.relu(output)
  
  with nn.parameter_scope("conv3"):
    output = PF.convolution(output, 64, (3, 3), stride=(1, 1), pad=(1, 1))
    output = F.relu(output)

  output = output.reshape([BATCH_SIZE, int(output.size / BATCH_SIZE)])

  with nn.parameter_scope("fc1"):
    output = PF.affine(output, 1024)
    output = F.relu(output)
  
  with nn.parameter_scope("fc2"):
    output = PF.affine(output, 256)
    output = F.relu(output)
  
  with nn.parameter_scope("softmax"):
    output = PF.affine(output, 10)
    output = F.softmax(output)
  
  return output


def loss(p, t):
  return F.mean(F.sum(F.sigmoid_cross_entropy(p, t), axis=1))


def accuracy(p, t):
  pred_and_label = [(np.argmax(_p), np.argmax(_t)) for _p, _t in zip(p.d, t.d)]
  return float(len(filter(lambda x: x[0] == x[1], pred_and_label))) / float(len(p.d))

if __name__ == '__main__':
  main()