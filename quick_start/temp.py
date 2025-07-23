#
# # __setattr__
# class Base:
#     def __setattr__(self, name, value):
#         print(f"Base setattr: {name}")
#         super().__setattr__(name, value)
#
# class Child(Base):
#     def __setattr__(self, name, value):
#         print(f"Child setattr: {name}")
#         super().__setattr__(name, value)  # вызовет Base.__setattr__, который снова вызывает super...
#
# ch = Child()
# ch.x = 123
#
#
# # AutoGrad
# import torch
#
# a = torch.tensor([[3., 3.],
#                   [3., 3.]], requires_grad=True)
#
# print(torch.is_grad_enabled())
#
# with torch.no_grad():
#     print(torch.is_grad_enabled())
#
# b = a * 4 # 12
# # PyTorch save gradients only for leaf nodes, by default, i.e. a which we created ourselves.
# # It saves memory by freeing computation graph after backward().
# # Therefore we have to use retain_grad().
# b.retain_grad() # in-place operation, like requires_grad_
#
# c = b + 3 # 15
# c.retain_grad() # in-place operation, like requires_grad_
#
# d = (a + 2) # 5
# d.retain_grad() # in-place operation, like requires_grad_
#
# e = c * d # 15 * 5
# e.retain_grad() # in-place operation, like requires_grad_
#
# e_sum = e.sum() # 75
# e_sum.retain_grad() # in-place operation, like requires_grad_
#
# e_sum.backward()
#
# print("e_sum:", e_sum.grad)
# print("e:", e.grad)
# print("d:", d.grad) # c
# print("c:", c.grad) # d
# print("b_grad:", b.grad)
# print("a_grad:", a.grad)
#
# print(torch.is_grad_enabled())
#
#
# # with
# class MyContext:
#     def __enter__(self):
#         print(">> Входим в блок with")
#         return 123
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         print("<< Выходим из блока with")
#         if exc_type:
#             print(f"Произошло исключение: {exc_type.__name__} - {exc_val}")
#         # Если True — подавит исключение, False — пробросит
#         return False
#
# # Используем with
# with MyContext() as value:
#     print(f"Внутри блока: {value}")
#     raise ValueError("Тестовая ошибка")  # <-- ошибка
#     print("Никогда не выполнится")
#
# print("После блока with")


import torch

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.devices import print_available_devices
# print(sys.path)

def lol():
    print_available_devices()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
