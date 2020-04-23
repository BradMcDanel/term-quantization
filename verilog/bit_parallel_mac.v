`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/25/2019 12:42:22 PM
// Design Name: 
// Module Name: bit_parallel_mac
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


module bit_parallel_mac #(
  parameter WEIGHT_RESET_VAL = 8'b00100011,
  parameter SHARED_W         = 0
)(
  input             clk               ,
  input             reset             ,
  input             update_w          ,
  input             mac_en            ,
  input      [7:0]  x, weight         ,
  input      [31:0] accumulation      ,
  output reg [31:0]       result            
);
  
  reg [7:0]  W;
  generate if(SHARED_W == 0) begin: g_W
      always @(posedge clk) begin : proc_W
        if(reset) begin
          W <= WEIGHT_RESET_VAL;
        end else if(update_w) begin
          W <= weight;
        end
      end
    end
    else begin
      always @(*) begin : proc_W
        W = weight;
      end
    end
  endgenerate
  
   always@(posedge clk) begin
    if(reset) begin
      result       <= 0;
    end
    else if(mac_en) begin
      result       <= accumulation + x*weight;
    end
  end
endmodule
