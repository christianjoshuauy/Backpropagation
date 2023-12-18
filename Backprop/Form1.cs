using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Backprop
{
    public partial class Form1 : Form
    {
        NeuralNet nn;
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            nn = new NeuralNet(4, 63, 1);
        }

        private void button2_Click(object sender, EventArgs e)
        {

            for (int x = 0; x < 110; x++)
            {
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        for (int k = 0; k < 2; k++)
                            for (int l = 0; l < 2; l++)
                            {
                                nn.setInputs(0, Convert.ToDouble(i));
                                nn.setInputs(1, Convert.ToDouble(j));
                                nn.setInputs(2, Convert.ToDouble(k));
                                nn.setInputs(3, Convert.ToDouble(l));
                                double desiredOutp = 0;
                                if(i == 1 && j == 1 && k == 1 && l == 1) desiredOutp = 1;
                                nn.setDesiredOutput(0, desiredOutp);
                                nn.learn();
                            }
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            nn.setInputs(0, Convert.ToDouble(textBox1.Text));
            nn.setInputs(1, Convert.ToDouble(textBox2.Text));
            nn.setInputs(2, Convert.ToDouble(textBox4.Text));
            nn.setInputs(3, Convert.ToDouble(textBox5.Text));
            nn.run();
            textBox3.Text = "" + nn.getOuputData(0);
        }

        private double CalculateError(NeuralNet nn)
        {
            int numSamples = 16;
            double totalError = 0.0;

            double[][] trainingData = new double[][] {
                new double[] { 0.0, 0.0, 0.0, 0.0 },
                new double[] { 0.0, 0.0, 0.0, 1.0 },
                new double[] { 0.0, 0.0, 1.0, 0.0 },
                new double[] { 0.0, 0.0, 1.0, 1.0 },
                new double[] { 0.0, 1.0, 0.0, 0.0 },
                new double[] { 0.0, 1.0, 0.0, 1.0 },
                new double[] { 0.0, 1.0, 1.0, 0.0 },
                new double[] { 0.0, 1.0, 1.0, 1.0 },
                new double[] { 1.0, 0.0, 0.0, 0.0 },
                new double[] { 1.0, 0.0, 0.0, 1.0 },
                new double[] { 1.0, 0.0, 1.0, 0.0 },
                new double[] { 1.0, 0.0, 1.0, 1.0 },
                new double[] { 1.0, 1.0, 0.0, 0.0 },
                new double[] { 1.0, 1.0, 0.0, 1.0 },
                new double[] { 1.0, 1.0, 1.0, 0.0 },
                new double[] { 1.0, 1.0, 1.0, 1.0 }
            };

            double[] desiredOutputs = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

            for (int i = 0; i < numSamples; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    nn.setInputs(j, trainingData[i][j]);
                }

                nn.run();

                double predictedOutput = nn.getOuputData(0);

                double error = 0.5 * Math.Pow(desiredOutputs[i] - predictedOutput, 2);

                totalError += error;
            }

            double meanError = totalError / numSamples;

            return meanError;
        }


        private Tuple<int, int> FindMinimumHiddenNeuronsAndEpochs()
        {
            int minHiddenNeurons = 0;
            int minEpochs = 0;
            double bestError = double.MaxValue;

            for (int hiddenNeurons = 1; hiddenNeurons <= 64; hiddenNeurons++)
            {
                for (int epochs = 10; epochs <= 1000; epochs += 10)
                {
                    NeuralNet nn = new NeuralNet(4, hiddenNeurons, 1);
                    double lastError = double.MaxValue;

                    for (int epoch = 0; epoch < epochs; epoch++)
                    {
                        for(int i = 0; i < 2; i++)
                            for(int j = 0; j < 2; j++)
                                for(int k = 0; k < 2; k++)
                                    for(int l = 0; l < 2; l++)
                                    {
                                        nn.setInputs(0, Convert.ToDouble(i));
                                        nn.setInputs(1, Convert.ToDouble(j));
                                        nn.setInputs(2, Convert.ToDouble(k));
                                        nn.setInputs(3, Convert.ToDouble(l));
                                        double desiredOutp = 0;
                                        if (i == 1 && j == 1 && k == 1 && l == 1) desiredOutp = 1;
                                        nn.setDesiredOutput(0, desiredOutp);
                                        nn.learn();
                                    }

                        double error = CalculateError(nn);

                        if (Math.Abs(error - lastError) < 0.001)
                        {
                            break;
                        }

                        lastError = error;
                    }

                    double finalError = CalculateError(nn);

                    if (finalError < bestError)
                    {
                        bestError = finalError;
                        minHiddenNeurons = hiddenNeurons;
                        minEpochs = epochs;
                    }
                }
            }

            return new Tuple<int, int>(minHiddenNeurons, minEpochs);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Tuple<int, int> res = FindMinimumHiddenNeuronsAndEpochs();
            MessageBox.Show("Min Hidden Neurons: " + res.Item1.ToString() + " | Min Epochs: " + res.Item2.ToString());
        }
    }
}
