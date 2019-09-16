
Arc=('GRU' 'LSTM')
LearningCurr=('DeCu' 'VoCu' 'NaCu')
Instance=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
Mode=('Init' 'Control' 'SlowPoints' 'CenterMass')



for q in {0..3}
do
    for k in {0..4}
    do
        for j in {0..1}
        do
            for i in {0..1}
            do
                python3 TriggerCIFARSpeedCont.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]} ${Mode[q]}
                python3 BlackBoxCIFAR.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]} ${Mode[q]}
            done
        done
    done
done

for q in {0..3}
do
    for k in {0..4}
    do
        for j in {0..1}
        do
            for i in {0..1}
            do
                python3 AccCIFAR.py ${Arc[i]} ${LearningCurr[j]} ${Instance[k]} ${Mode[q]}
            done
        done
    done
done
