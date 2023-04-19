def kristianskonto(n):
    if n == 1:
            return 50000*1.02
    return 50000*1.02**n + kristianskonto(n-1)
a = 7
print(kristianskonto(a))


