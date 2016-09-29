#!/usr/bin/zsh -e

MODEL=$1
DECODE=$2
#INDEX=$3
#DATA=/path/to/agiga_work/valid.article.filter.txt
#INPUT=`sed -n "${INDEX}p" $DATA`
EXAMPLE1="five-time world champion michelle kwan withdrew from the #### us figure skating championships on wednesday , but will petition us skating officials for the chance to compete at the #### turin olympics ."
EXAMPLE2="at least two people have tested positive for the bird flu virus in eastern turkey , health minister recep akdag told a news conference wednesday ."
INPUT=$EXAMPLE1

echo 'INPUT:'
echo $INPUT
echo 

echo "${MODEL}"
for LEN in 30 50 75
do
  out=`python predict.py ${MODEL} =(echo $INPUT) ${LEN} -d $DECODE --no_truncate`
  echo ${LEN}"\t"`echo $out | wc -c`"\t"`echo $out`
done
