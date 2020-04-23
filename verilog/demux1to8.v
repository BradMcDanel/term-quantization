`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/08/2018 08:48:35 AM
// Design Name: 
// Module Name: demux1to9
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

module demux1to8
(
    clk,
    sel,
    reset,
    Data_out 
    );
    input clk;
    input reset;
    input [2:0] sel;
    output reg [7:0] Data_out;

//always block with Data_in and sel in its sensitivity list
always @ (posedge clk) begin
    if (reset) begin
        Data_out <= 'b0;
    end
    else begin
        case (sel)
            3'b000 : begin
                Data_out[0] <= 'b1;
            end
            3'b001 : begin
                Data_out[1] <= 'b1;
            end
            3'b010 : begin
                Data_out[2] <= 'b1;
            end
            3'b011 : begin
                Data_out[3] <= 'b1;
            end
            3'b100 : begin
                Data_out[4] <= 'b1;
            end
            3'b101 : begin
                Data_out[5] <= 'b1;
            end
            3'b110 : begin
                Data_out[6] <= 'b1; 
            end                  
            3'b111 : begin     
                Data_out[7] <= 'b1; 
            end                  
            default : begin
                Data_out <= 'b0;
            end
        endcase
    end
    end
    endmodule


