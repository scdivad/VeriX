import time
for i in range(2):
	print(i)
	time.sleep(1)
	if i == 3: print(0/0)
