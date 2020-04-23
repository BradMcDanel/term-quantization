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
  parameter NUM_COE_ARRAY   = 16,
  parameter NUM_COMBINED_TERMS = 8,
  parameter NUM_BIT_EXPONENT = 3
)(
  input                            clk            ,
  input                            reset          ,
  input                 [  NUM_COE_ARRAY-1:0] accumulation   ,   //16
  input   [NUM_BIT_EXPONENT * NUM_COMBINED_TERMS-1:0] input_exponent,   // 4*8 = 32
  input   [NUM_COMBINED_TERMS-1:0]       sign_ctrl,
  input  [   3:0] data_terms,
  input  [   4:0] group_size,
  input  [   6:0] group_budget,
  input   start_shift                             , // this it the signal controls the saving of the input registers
  output              [  NUM_COE_ARRAY-1:0] output_result
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
    
  reg [NUM_BIT_EXPONENT * NUM_COMBINED_TERMS-1:0] input_exponent_reg;
  reg [NUM_BIT_EXPONENT * NUM_COMBINED_TERMS-1:0] weight_reg;
  reg [NUM_BIT_EXPONENT-1 : 0] exponents, weight;
  reg [clog2(NUM_COMBINED_TERMS)-1 :0] counter;
  reg [NUM_COMBINED_TERMS-1:0] sign_ctrl_reg;
  wire [NUM_BIT_EXPONENT:0] input_selection;
  wire [NUM_COMBINED_TERMS-1:0] accumulation_in [NUM_COE_ARRAY-1:0];
  wire [NUM_COMBINED_TERMS-1:0] accumulation_out [NUM_COE_ARRAY-1:0];
  
  assign input_selection = exponents + weight;
  
  always@(posedge clk) begin
      if(reset == 1) begin
          counter <= 'b0;
          input_exponent_reg <= 'b0;
      end else if (start_shift == 1'b1) begin   
          input_exponent_reg[NUM_BIT_EXPONENT * NUM_COMBINED_TERMS-1:0] <= input_exponent[NUM_BIT_EXPONENT * NUM_COMBINED_TERMS-1:0];
          sign_ctrl_reg[NUM_COMBINED_TERMS-1:0] <= sign_ctrl[NUM_COMBINED_TERMS-1:0];
      end  
  end
  
  always@(posedge clk) begin
        if(reset == 1) begin
            weight_reg <= 'b0;
            counter <= 'b0;
        end else begin
            counter <= counter + 1'b1; 
            exponents[0] <= input_exponent_reg[counter*data_terms];
            exponents[1] <= input_exponent_reg[counter*data_terms+1];
            exponents[2] <= input_exponent_reg[counter*data_terms+2];
            weight[0] <= weight_reg[counter*data_terms];
            weight[1] <= weight_reg[counter*data_terms+1];
            weight[2] <= weight_reg[counter*data_terms+2];
        end
    end  

  genvar i;
  assign accumulation_in[0] = accumulation;
  assign output_result = accumulation_out[NUM_COMBINED_TERMS-1] ;
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
