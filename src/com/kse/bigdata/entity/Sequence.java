package com.kse.bigdata.entity;

import java.io.IOException;
import java.util.LinkedList;

/**
 * Created by bk on 14. 12. 3.
 */
public class Sequence implements Comparable<Sequence>{

        public static final int SIZE_OF_SEQUENCE = 36;
        public static final int SIZE_OF_HEAD_SEQ = 6;
        public static final int SIZE_OF_TAIL_SEQ = 30;

        private double euclideanDistance = 100.0d;
        private double[] sequence = new double[SIZE_OF_SEQUENCE];
        private double[] head     = new double[SIZE_OF_HEAD_SEQ];
        private double[] tail     = new double[SIZE_OF_TAIL_SEQ];
        private double[] normTail = null;

        public Sequence(String totalInput) throws IOException {
            if(totalInput.equals(""))
                throw new IOException();

            parseStringToSequence(totalInput);
        }

        public Sequence(LinkedList<Double> inputSequence){
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = inputSequence.get(index);
                sequence[index] = value;

                if(index < SIZE_OF_HEAD_SEQ) {
                    head[index] = value;
                } else{
                    tail[index - SIZE_OF_HEAD_SEQ] = value;
                }
            }
        }

        public double getEuclideanDistance() { return this.euclideanDistance; }

        public void setEuclideanDistance(double distance){ this.euclideanDistance = distance; }

        public void setNormTail(double[] normTail) { this.normTail = normTail; }

        /**
         * Parse the sequence data to string.<br>
         * {1,2,3,4,5} will be parsed to "1-2-3-4-5".
         * @return
         */
        @Override
        public String toString(){
            StringBuilder stringBuilder = new StringBuilder();
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                stringBuilder.append(String.valueOf(sequence[index]));
                stringBuilder.append("-");
            }

            stringBuilder.append(euclideanDistance);

            return stringBuilder.toString();
        }

        /**
         * Parse string to sequence.<br>
         * "1-2-3-4-5" will be parsed to {1,2,3,4,5}.
         * @return
         */
        public void parseStringToSequence(String input){
            String[] values = input.split("-");
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = Double.valueOf(values[index]);
                sequence[index] = value;

                if(index < SIZE_OF_HEAD_SEQ) {
                    head[index] = value;
                } else {
                    tail[index - SIZE_OF_HEAD_SEQ] = value;
                }
            }

            this.euclideanDistance = Double.valueOf(values[SIZE_OF_SEQUENCE]);
        }


        public double[] getHead(){
            return this.head;
        }

        public double[] getTail(boolean isNormalized){
            if(isNormalized)
                return this.normTail;

            return this.tail;
        }

        @Override
        public int hashCode() {
            return sequence.hashCode();
        }

        @Override
        public boolean equals(Object obj) {
            if(this == obj)
                return true;

            if(obj instanceof Sequence) {

                Sequence seq = ((Sequence) obj);

                for (int index = 0; index < SIZE_OF_SEQUENCE; index++)
                    if (seq.sequence[index] != this.sequence[index])
                        return false;

                return true;

            } else {
                return false;

            }
        }

        @Override
        public int compareTo(Sequence o) {
            return Double.compare(this.euclideanDistance, o.euclideanDistance);
        }

}