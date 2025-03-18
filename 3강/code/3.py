import torch
import numpy as np

print(torch.cuda.is_available())

# tc = torch.tensor([[1., -1.], [1., -1.]])
# print(tc)
#
# print(torch.tensor([[1, 2], [3, 4]]))
# print(torch.tensor([[1, 2], [3, 4]], device='cuda:0'))
# print(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64))
#
# v = torch.tensor([1, 2, 3])
# w = torch.tensor([3, 4, 6])
# print(w - v)
#
# temp = torch.tensor([[1, 2], [3, 4]])
# print(temp.shape)
# print('-----------')
# print(temp.size())
# print('-----------')
# print(temp.view(4, 1)) # 2*2 를 4*1 로 변형
# print('-----------')
# print(temp.view(-1)) # 2*2 를 1차원 벡터로
# print('-----------')
# print(temp.view(1, -1)) # (1, -1) 은 (1, ?) 와 같은 의미, 원소개수 2*2=4 를 유지한채 (1,?) 의 형태를 만든다는것. -> (1,4)
# print('-----------')
# print(temp.view(-1, 1)) # (-1, 1) 은 (?, 1) 와 같은 의미, 원소개수 2*2=4 를 유지한채 (?,1) 의 형태를 만든다는것. -> (4,1)


# x = torch.empty(5, 4)  # 5*4의 텐서를 생성, 초기화되지 않는 메모리가 그대로 보여짐
# print(x)

# ones = torch.ones(3, 3)  # 3*3의 텐서를 생성, 1로 초기화
# print(ones)

# zeros = torch.zeros(2)  # 2*1의 텐서를 생성, 0로 초기화
# print(zeros)

# zeros2 = torch.zeros(2, 2)  # 2*2의 텐서를 생성, 0로 초기화
# print(zeros2)

# rand = torch.rand(5, 6)  # 5*6의 텐서를 생성, 랜덤값으로 초기화
# print(rand)

# 리스트 넘파이 배열 텐서로 만듫기
# l = [13, 4]
# n = np.array([4, 55, 7])
# print(torch.tensor(l))
# print(torch.tensor(n))

# x = torch.zeros(5,4)
# print(x.size())
# print(x.size()[1])

# x = torch.rand(2,2)
# y=torch.rand(2,2)
# print(x)
# print(y)
# # print(x+y)
# print('------------')
# print(torch.add(x,y))
# print(y.add(x))
# print(y)

# 텐서의 크기 변환
x=torch.rand(8,8)
print(x.size())
print('------------')
a=x.view(64)
print(a.size())
print('------------')
b=x.view(-1,4,4)
print(b.size())
## 16:27
