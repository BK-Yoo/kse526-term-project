package com.kse.bigdata.dist;

import com.kse.bigdata.entity.Sequence;
import org.apache.commons.math3.util.FastMath;

/**
 * Created by bk on 14. 12. 9.
 * kse 526 group assignment.
 */
public class DTW {

    private final double[] actualSeq;
    private final double BIG_VALUE = Double.MAX_VALUE;

    private double cost;
    private final int DTW_WIDTH = Sequence.SIZE_OF_WINDOW + 1;
    private double[][] DTW = new double[DTW_WIDTH][DTW_WIDTH];


    public DTW(double[] actualSeq){
        this.actualSeq = actualSeq;
        for(int i = 1; i < DTW_WIDTH; i++){
            DTW[i][0] = BIG_VALUE;
            DTW[0][i] = BIG_VALUE;
        }
    }

    public double evaluate(double[] comparison){
        clear();

        for(int i = 1; i < DTW_WIDTH; i++){
            for(int j = 1; j < DTW_WIDTH; j++){
                cost = FastMath.abs(actualSeq[i-1] - comparison[j-1]);
                DTW[i][j] = cost + getMin(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1]);
            }
        }

        return DTW[DTW_WIDTH - 1][DTW_WIDTH - 1];
    }

    private void clear(){
        for(int i = 1; i < DTW_WIDTH; i++) {
            for (int j = 1; j < DTW_WIDTH; j++) {
                DTW[i][j] = 0.0d;
            }
        }
    }

    private double getMin(double a, double b, double c){
        double second = (a < b)? a : b;
        return (second < c)? second : c;
    }
}
