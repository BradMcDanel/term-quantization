`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/01/2019 04:42:34 PM
// Design Name: 
// Module Name: coe_acc
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


module coe_acc #(
  parameter NUM_COE_ARRAY   = 16,
  parameter INPUT_SEL_WIDTH = 4
  // log2(NUM_COE_ARRAY) must less or equal to INPUT_SEL_WIDTH
) (
  input                            clk            ,
  input                            reset          ,
  input      [  NUM_COE_ARRAY-1:0] accumulation   ,
  input      [INPUT_SEL_WIDTH-1:0] input_selection,
  input                            sign_ctrl      ,
  output     [  NUM_COE_ARRAY-1:0] result
);

// define all the register and wires
reg [INPUT_SEL_WIDTH-1:0] input_selection_reg;
wire select_acc;
wire adder_results;
wire [NUM_COE_ARRAY-1:0] data_out;

// call the bit serial adder
bit_serial_adder  adder (.clk(clk),.reset(reset),.input_stream(accumulation[input_selection_reg]),.sign_ctrl(sign_ctrl),.adder_results(adder_results));
demux  de_mux (.clk(clk),.reset(reset),.sel(input_selection),.Data_out(data_out));

assign select_acc = accumulation[input_selection_reg];
assign result = ((~(data_out)&accumulation)|(data_out)&{(NUM_COE_ARRAY){adder_results}});

endmodule

    