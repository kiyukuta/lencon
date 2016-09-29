#!/usr/bin/zsh -e

DATA=/path/to/clean_2004/input.txt # please see https://github.com/facebook/NAMAS#constructing-the-data-set
BASE=$1
mkdir -p ${BASE}


function FIXLEN() {
  M=$1
  L=$2
  B=$3
  O=$4
  python predict.py ${M} ${DATA} ${L} -d bs_fixlen -b ${B} --log ${O}/jsons
}


function FIXRNG() {
  M=$1
  L=$2
  B=$3
  O=$4
  MINL=$5
  python predict.py ${M} ${DATA} ${L} -d bs_fixrng --min_length ${MINL} -b ${B} --log ${O}/jsons
}


function BEAM_S() {
  M=$1
  L=$2
  B=$3
  O=$4
  python predict.py ${M} ${DATA} ${L} -d bs_normal -b ${B} --log ${O}/jsons
}


TRAINED_MODELS=../models/

for LEN in 30 50 75
do
  echo  $LEN
  OUT=${BASE}/${LEN}
  mkdir -p ${OUT}
  mkdir -p ${OUT}/jsons
  mkdir -p ${OUT}/system
  
  FIXLEN ${TRAINED_MODELS}/encdec  ${LEN} 10 ${OUT}              > ${OUT}/system/task1_fixlen &
  FIXRNG ${TRAINED_MODELS}/encdec  ${LEN} 30 ${OUT} $((LEN - 5)) > ${OUT}/system/task1_fixrng &
  FIXRNG ${TRAINED_MODELS}/lenemb  ${LEN} 10 ${OUT} 0            > ${OUT}/system/task1_lenemb &
  FIXRNG ${TRAINED_MODELS}/leninit ${LEN} 10 ${OUT} 0            > ${OUT}/system/task1_leninit &
  wait

  BEAM_S ${TRAINED_MODELS}/lenemb  ${LEN} 10 ${OUT}              > ${OUT}/system/task1_lenemb_inf  &
  BEAM_S ${TRAINED_MODELS}/leninit ${LEN} 10 ${OUT}              > ${OUT}/system/task1_leninit_inf &
  wait

done



