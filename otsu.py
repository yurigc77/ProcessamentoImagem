def otsu(img):
    sum = 0
    sumB = 0
    wB = 0 #peso do fundo
    wF = 0 #peso da frente
    varMax = 0
    threshold = 0 #limite da limiarização

    hist = his.histogramaOtsu(img)#faz o histograma da imagem
    total = img.shape[0] * img.shape[1]#numero de linhas x colunas (total de pixels)

    for i in range(0, 256):
        sum += i * hist[i]
        


    for i in range(0, 256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB #peso da frente é o total de pixels menos o peso do fundo
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB #calculando media do fundo
        mF = (sum - sumB) / wF #calculando media da frente
        varBetween = wB * wF * (mB - mF) * (mB - mF) #calculo da variancia
        if varBetween > varMax:#a variancia que for maior que max determina o ton de limearização
            varMax = varBetween
            threshold = i#valor limite

    return threshold #retorna o limite



    #(thresh, bin) = cv.threshold(filtrada, otsu(gray), 255, cv.THRESH_BINARY) #binarizando com otsu como limite
    #cv.imshow('Binarizada Otsu', bin)


    #(thresh, bin) = cv.threshold(filtrada, 120, 255, cv.THRESH_BINARY)#binarização com limite fixo para comparar
    #cv.imshow('Binarizada Fixa', bin)

    #plt.show()#mostra o histograma