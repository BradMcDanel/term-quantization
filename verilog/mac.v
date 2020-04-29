`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 08/07/2019 01:57:02 PM
// Design Name:
// Module Name: mac
// Project Name:
// Target Devices:
// Tool Versions:
// Description:
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////

// mac assumes only input the useful items
module mac #(
  parameter NUM_COE_ARRAY      = 16,
  parameter NUM_COMBINED_TERMS = 8 ,
  parameter NUM_BIT_EXPONENT   = 3
) (
  input                                            clk           ,
  input                                            reset         ,
  input  [                      NUM_COE_ARRAY-1:0] accumulation  , //16
  input  [NUM_BIT_EXPONENT*NUM_COMBINED_TERMS-1:0] input_exponent, // 4*8 = 32
  input  [                 NUM_COMBINED_TERMS-1:0] sign_ctrl     ,
  input  [                                    3:0] data_terms    ,
  input  [                                    4:0] group_size    ,
  input  [                                    6:0] group_budget  ,
  input                                            start_shift   , // this it the signal controls the saving of the input registers
  output [                      NUM_COE_ARRAY-1:0] output_result
);

  function [31:0] clog2 (input [31:0] x);
    reg [31:0] x_tmp;
    begin
      x_tmp = x-1;
      for(clog2=0; x_tmp>0; clog2=clog2+1) begin
        x_tmp = x_tmp >> 1;
      end
    end
  endfunction

  reg  [NUM_BIT_EXPONENT*NUM_COMBINED_TERMS-1:0]  input_exponent_reg                   ;
  reg  [NUM_BIT_EXPONENT*NUM_COMBINED_TERMS-1:0]  weight_reg                           ;
  reg  [                   NUM_BIT_EXPONENT-1:0]  exponents, weight;
  reg  [          clog2(NUM_COMBINED_TERMS)-1:0]  counter                              ;
  reg  [                 NUM_COMBINED_TERMS-1:0]  sign_ctrl_reg                        ;
  wire [                     NUM_BIT_EXPONENT:0]  input_selection                      ;
  wire [                 NUM_COMBINED_TERMS-1:0]  accumulation_in   [NUM_COE_ARRAY-1:0];
  wire [                 NUM_COMBINED_TERMS-1:0]  accumulation_out  [NUM_COE_ARRAY-1:0];
  reg  [                                    4:0]  group_size_reg                       ;
  reg  [                                    6:0]  group_budget_reg                     ;
  assign input_selection = exponents + weight;

  always@(posedge clk) begin
    if(reset == 1) begin
      counter            <= 'b0;
      input_exponent_reg <= 'b0;
      group_size_reg     <= 'b0;
      group_budget_reg   <= 'b0;
    end else if (start_shift == 1'b1) begin
      group_size_reg                                         <= group_size;
      group_budget_reg                                       <= group_budget;
      input_exponent_reg[group_budget_reg*group_size_reg]    <= input_exponent[group_budget_reg * group_size_reg];
      input_exponent_reg[group_budget_reg*group_size_reg-1]  <= input_exponent[group_budget_reg * group_size_reg-1];
      input_exponent_reg[group_budget_reg*group_size_reg-2]  <= input_exponent[group_budget_reg * group_size_reg-2];
      input_exponent_reg[group_budget_reg*group_size_reg-3]  <= input_exponent[group_budget_reg * group_size_reg-3];
      input_exponent_reg[group_budget_reg*group_size_reg-4]  <= input_exponent[group_budget_reg * group_size_reg-4];
      input_exponent_reg[group_budget_reg*group_size_reg-5]  <= input_exponent[group_budget_reg * group_size_reg-5];
      input_exponent_reg[group_budget_reg*group_size_reg-6]  <= input_exponent[group_budget_reg * group_size_reg-6];
      input_exponent_reg[group_budget_reg*group_size_reg-7]  <= input_exponent[group_budget_reg * group_size_reg-7];
      input_exponent_reg[group_budget_reg*group_size_reg-8]  <= input_exponent[group_budget_reg * group_size_reg-8];
      input_exponent_reg[group_budget_reg*group_size_reg-9]  <= input_exponent[group_budget_reg * group_size_reg-9];
      input_exponent_reg[group_budget_reg*group_size_reg-10] <= input_exponent[group_budget_reg * group_size_reg-10];
      input_exponent_reg[group_budget_reg*group_size_reg-11] <= input_exponent[group_budget_reg * group_size_reg-11];
      input_exponent_reg[group_budget_reg*group_size_reg-12] <= input_exponent[group_budget_reg * group_size_reg-12];
      input_exponent_reg[group_budget_reg*group_size_reg-13] <= input_exponent[group_budget_reg * group_size_reg-13];
      input_exponent_reg[group_budget_reg*group_size_reg-14] <= input_exponent[group_budget_reg * group_size_reg-14];
      input_exponent_reg[group_budget_reg*group_size_reg-15] <= input_exponent[group_budget_reg * group_size_reg-15];
      input_exponent_reg[group_budget_reg*group_size_reg-16] <= input_exponent[group_budget_reg * group_size_reg-16];
      input_exponent_reg[group_budget_reg*group_size_reg-17] <= input_exponent[group_budget_reg * group_size_reg-17];
      input_exponent_reg[group_budget_reg*group_size_reg-18] <= input_exponent[group_budget_reg * group_size_reg-18];
      input_exponent_reg[group_budget_reg*group_size_reg-19] <= input_exponent[group_budget_reg * group_size_reg-19];
      input_exponent_reg[group_budget_reg*group_size_reg-20] <= input_exponent[group_budget_reg * group_size_reg-20];
      input_exponent_reg[group_budget_reg*group_size_reg-21] <= input_exponent[group_budget_reg * group_size_reg-21];
      input_exponent_reg[group_budget_reg*group_size_reg-22] <= input_exponent[group_budget_reg * group_size_reg-22];
      input_exponent_reg[group_budget_reg*group_size_reg-23] <= input_exponent[group_budget_reg * group_size_reg-23];


      sign_ctrl_reg[group_budget_reg*group_size_reg]    <= sign_ctrl[group_budget_reg * group_size_reg];
      sign_ctrl_reg[group_budget_reg*group_size_reg-1]  <= sign_ctrl[group_budget_reg * group_size_reg-1];
      sign_ctrl_reg[group_budget_reg*group_size_reg-2]  <= sign_ctrl[group_budget_reg * group_size_reg-2];
      sign_ctrl_reg[group_budget_reg*group_size_reg-3]  <= sign_ctrl[group_budget_reg * group_size_reg-3];
      sign_ctrl_reg[group_budget_reg*group_size_reg-4]  <= sign_ctrl[group_budget_reg * group_size_reg-4];
      sign_ctrl_reg[group_budget_reg*group_size_reg-5]  <= sign_ctrl[group_budget_reg * group_size_reg-5];
      sign_ctrl_reg[group_budget_reg*group_size_reg-6]  <= sign_ctrl[group_budget_reg * group_size_reg-6];
      sign_ctrl_reg[group_budget_reg*group_size_reg-7]  <= sign_ctrl[group_budget_reg * group_size_reg-7];
      sign_ctrl_reg[group_budget_reg*group_size_reg-8]  <= sign_ctrl[group_budget_reg * group_size_reg-8];
      sign_ctrl_reg[group_budget_reg*group_size_reg-9]  <= sign_ctrl[group_budget_reg * group_size_reg-9];
      sign_ctrl_reg[group_budget_reg*group_size_reg-10] <= sign_ctrl[group_budget_reg * group_size_reg-10];
      sign_ctrl_reg[group_budget_reg*group_size_reg-11] <= sign_ctrl[group_budget_reg * group_size_reg-11];
      sign_ctrl_reg[group_budget_reg*group_size_reg-12] <= sign_ctrl[group_budget_reg * group_size_reg-12];
      sign_ctrl_reg[group_budget_reg*group_size_reg-13] <= sign_ctrl[group_budget_reg * group_size_reg-13];
      sign_ctrl_reg[group_budget_reg*group_size_reg-14] <= sign_ctrl[group_budget_reg * group_size_reg-14];
      sign_ctrl_reg[group_budget_reg*group_size_reg-15] <= sign_ctrl[group_budget_reg * group_size_reg-15];
      sign_ctrl_reg[group_budget_reg*group_size_reg-16] <= sign_ctrl[group_budget_reg * group_size_reg-16];
      sign_ctrl_reg[group_budget_reg*group_size_reg-17] <= sign_ctrl[group_budget_reg * group_size_reg-17];
      sign_ctrl_reg[group_budget_reg*group_size_reg-18] <= sign_ctrl[group_budget_reg * group_size_reg-18];
      sign_ctrl_reg[group_budget_reg*group_size_reg-19] <= sign_ctrl[group_budget_reg * group_size_reg-19];
      sign_ctrl_reg[group_budget_reg*group_size_reg-20] <= sign_ctrl[group_budget_reg * group_size_reg-20];
      sign_ctrl_reg[group_budget_reg*group_size_reg-21] <= sign_ctrl[group_budget_reg * group_size_reg-21];
      sign_ctrl_reg[group_budget_reg*group_size_reg-22] <= sign_ctrl[group_budget_reg * group_size_reg-22];
      sign_ctrl_reg[group_budget_reg*group_size_reg-23] <= sign_ctrl[group_budget_reg * group_size_reg-23];
      //sign_ctrl_reg[NUM_COMBINED_TERMS-1:0] <= sign_ctrl[NUM_COMBINED_TERMS-1:0];
    end
  end

  always@(posedge clk) begin
    if(reset == 1) begin
      weight_reg <= 'b0;
      counter    <= 'b0;
    end else begin
      counter      <= counter + 1'b1;
      exponents[0] <= input_exponent_reg[counter*data_terms];
      exponents[1] <= input_exponent_reg[counter*data_terms+1];
      exponents[2] <= input_exponent_reg[counter*data_terms+2];
      weight[0]    <= weight_reg[counter*data_terms];
      weight[1]    <= weight_reg[counter*data_terms+1];
      weight[2]    <= weight_reg[counter*data_terms+2];
    end
  end

  genvar i;
  assign accumulation_in[0] = accumulation;
  assign output_result      = accumulation_out[NUM_COMBINED_TERMS-1] ;
  generate
    for (i=1; i<=8-1; i=i+1) begin
      begin
        assign accumulation_in[i] = accumulation_out[i-1];
      end
      begin : QUANTIZATION_BLOCK
        coe_acc#(
          .NUM_COE_ARRAY(NUM_COE_ARRAY),
          .INPUT_SEL_WIDTH(NUM_BIT_EXPONENT+1)
        )  pe (
          .clk         (clk                        ),
          .reset       (reset                      ),
          .accumulation(accumulation_in[i]         ),
          .input_selection  (input_selection       ),    // change here to count for the late start signal
          .sign_ctrl   (sign_ctrl_reg[i]               ),
          .result      (accumulation_out[i]        )
        );
      end
    end
  endgenerate

endmodule
