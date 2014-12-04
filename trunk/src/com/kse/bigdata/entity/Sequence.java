//        Copyright [BKYoo]
//
//        Licensed under the Apache License, Version 2.0 (the "License");
//        you may not use this file except in compliance with the License.
//        You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//        Unless required by applicable law or agreed to in writing, software
//        distributed under the License is distributed on an "AS IS" BASIS,
//        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//        See the License for the specific language governing permissions and
//        limitations under the License.

package com.kse.bigdata.entity;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * Created by bk on 14. 12. 3.
 * KSE526 Term Project
 */
public class Sequence implements Comparable<Sequence>{

        public static final int SIZE_OF_SEQUENCE = 36;
        public static final int SIZE_OF_HEAD_SEQ = 6;
        public static final int SIZE_OF_TAIL_SEQ = 30;

        private double euclideanDistance = 100.0d;
        private double[] head     = new double[SIZE_OF_HEAD_SEQ];
        private double[] tail     = new double[SIZE_OF_TAIL_SEQ];
        private double[] normHead = new double[SIZE_OF_HEAD_SEQ];

        public Sequence(String totalInput) throws IOException {
            if(totalInput.equals(""))
                throw new IOException();

            parseStringToSequence(totalInput);
        }

        public Sequence(LinkedList<Double> inputSequence){
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = inputSequence.get(index);
                if(index < SIZE_OF_HEAD_SEQ) {
                    head[index] = value;
                } else{
                    tail[index - SIZE_OF_HEAD_SEQ] = value;
                }
            }
        }

        public double getEuclideanDistance() { return this.euclideanDistance; }

        public void setEuclideanDistance(double distance){ this.euclideanDistance = distance; }

        /**
         * Parse the sequence data to string.<br>
         * {1,2,3,4,5} will be parsed to "1-2-3-4-5".
         * @return String "Sequence + Euclidean Dist"
         */
        @Override
        public String toString(){
            StringBuilder stringBuilder = new StringBuilder();
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                if(index < SIZE_OF_HEAD_SEQ) {
                    stringBuilder.append(String.valueOf(head[index]));
                } else {
                    stringBuilder.append(String.valueOf(tail[index - SIZE_OF_HEAD_SEQ]));
                }

                stringBuilder.append("-");
            }

            stringBuilder.append(euclideanDistance);

            return stringBuilder.toString();
        }

        /**
         * Parse string to sequence.<br>
         * "1-2-3-4-5" will be parsed to {1,2,3,4,5}.
         */
        public void parseStringToSequence(String input){
            String[] values = input.split("-");
            double value;
            for(int index = 0; index < SIZE_OF_SEQUENCE; index++) {
                value = Double.valueOf(values[index]);
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

        public double[] getNormHead(){
            return this.normHead;
        }

        public double[] getTail(){ return this.tail; }

        public String getTailString(){
            StringBuilder stringBuilder = new StringBuilder();
            for(int index = 0; index < SIZE_OF_TAIL_SEQ; index ++){
                stringBuilder.append(tail[index]);

                if(index == (SIZE_OF_TAIL_SEQ - 1))
                    break;

                stringBuilder.append("-");
            }

            return stringBuilder.toString();
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(head) + Arrays.hashCode(tail);
        }

        @Override
        public boolean equals(Object obj) {
            if(this == obj)
                return true;

            if(obj instanceof Sequence) {

                Sequence seq = ((Sequence) obj);

                for (int index = 0; index < SIZE_OF_SEQUENCE; index++){
                    if ((index < SIZE_OF_HEAD_SEQ)) {
                        if((seq.head[index] != this.head[index]))
                            return false;

                    } else if (seq.tail[index - SIZE_OF_HEAD_SEQ] == this.tail[index - SIZE_OF_HEAD_SEQ]) {
                        return false;
                    }
                }

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