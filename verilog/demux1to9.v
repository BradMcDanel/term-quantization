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

module demux1to9#(parameter CATCH_START_BIT = 10)
(
    clk,
    Data_in,
    sel,
    reset,
    Data_out_0,
    Data_out_1,
    Data_out_2,
    Data_out_3,
    Data_out_4,
    Data_out_5,
    Data_out_6,
    Data_out_7,
    Data_out_8   
    );
    input clk;
    input reset;
    input Data_in;
    input [4:0] sel;
    output reg Data_out_0;
    output reg Data_out_1;
    output reg Data_out_2;
    output reg Data_out_3;
    output reg Data_out_4;
    output reg Data_out_5;
    output reg Data_out_6;
    output reg Data_out_7;
    output reg Data_out_8;

//always block with Data_in and sel in its sensitivity list
always @ (posedge clk) begin
    if (reset) begin
        Data_out_0 <= 1'b0;
        Data_out_1 <= 1'b0;
        Data_out_2 <= 1'b0;
        Data_out_3 <= 1'b0;
        Data_out_4 <= 1'b0;
        Data_out_5 <= 1'b0;
        Data_out_6 <= 1'b0;
        Data_out_7 <= 1'b0;
        Data_out_8 <= 1'b0;
    end
    else begin
        case (sel)
            CATCH_START_BIT : begin
                Data_out_0 <= Data_in;
            end
            CATCH_START_BIT + 1 : begin
                Data_out_1 <= Data_in;
            end
            CATCH_START_BIT + 2 : begin
                Data_out_2 <= Data_in;
            end
            CATCH_START_BIT + 3 : begin
                Data_out_3 <= Data_in;
            end
            CATCH_START_BIT + 4 : begin
                Data_out_4 <= Data_in;
            end
            CATCH_START_BIT + 5 : begin
                Data_out_5 <= Data_in;
            end
            CATCH_START_BIT + 6 : begin
                Data_out_6 <= Data_in;
            end
            CATCH_START_BIT + 7 : begin
                Data_out_7 <= Data_in;
            end
            default : begin
                Data_out_8 <= Data_in;
            end
        endcase
    end
    end
    endmodule

