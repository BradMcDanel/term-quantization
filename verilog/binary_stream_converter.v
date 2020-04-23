`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 04/17/2020 09:00:26 PM
// Design Name:
// Module Name: binary_stream_converter
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


module binary_stream_converter #(
  parameter NUM_COE_ARRAY      = 16,
  parameter NUM_COMBINED_TERMS = 8
) (
  input                      clk            ,
  input                      reset          ,
  input  [NUM_COE_ARRAY-1:0] systolic_result,
  output                     binary_result
);
  reg [NUM_COMBINED_TERMS-1:0] buffer_reg0,buffer_reg1,buffer_reg2,buffer_reg3,buffer_reg4,buffer_reg5,buffer_reg6,buffer_reg7;
  reg [                  15:0] binary_reg ;
  always@(posedge clk) begin
    if(reset == 1) begin
      {buffer_reg0,buffer_reg1,buffer_reg2,buffer_reg3,buffer_reg4,buffer_reg5,buffer_reg6,buffer_reg7} <= 'b0;
    end begin
      buffer_reg0 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[0]};
      buffer_reg1 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[1]};
      buffer_reg2 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[2]};
      buffer_reg3 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[3]};
      buffer_reg4 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[4]};
      buffer_reg5 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[5]};
      buffer_reg6 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[6]};
      buffer_reg7 <= {buffer_reg0[NUM_COMBINED_TERMS-2:0], systolic_result[7]};
      binary_reg  <= 128 * buffer_reg7 + 64 * buffer_reg6 + 32 * buffer_reg5 + 16 * buffer_reg4 + 8 * buffer_reg3 + 4 * buffer_reg2 + 2 * buffer_reg1 + buffer_reg0;
    end
  end
  assign binary_result = binary_reg[15];

endmodule
