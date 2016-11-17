f = open('trained_set.txt')
content = f.read().split('\n')

for i in range(0, 5):
    items = content[2].split('|')
    list_data = []
    for item in items:
        criterias = item.split(',')
        list_data.append(str(criterias[i])+','+str(criterias[i+1]))
    list_data = '|'.join(list_data)
    print(list_data + '\n\n')
    f = open('TrainedData/' + str(i) + 'trained_set.txt', 'w')
    f.write('2' + '\n' + content[1] + '\n' + list_data + '\n' + content[3])
