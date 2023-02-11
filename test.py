
ls1 = [1,2]
ls2 = [4,5]

ls_comb = [ls1, ls2]

# c for b in a for c in b
#ls_split = [c for b in a for c in range(len(ls_comb))]
print(range(len(ls_comb)))
# Ikke sp'rg mig for hvor lang tid den her grimme satan tog at lave
ls_split = [a for i in range(len(ls_comb)) for a in [ls_comb[i][0], ls_comb[1][1]]]
print(ls_split)

for i in range(len(ls_comb)):
    for r in [ls_comb[i][0], ls_comb[i][1]]:
        print(r)

#
# list_of_words = []
# for sentence in text:
#     for word in sentence:
#        list_of_words.append(word)
# return list_of_words
#
# [word for sentence in text for word in sentence]

