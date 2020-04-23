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

module demux
(
    clk,
    sel,
    reset,
    Data_out 
    );
    input clk;
    input reset;
    input [4:0] sel;
    output reg [32-1:0] Data_out;

//always block with Data_in and sel in its sensitivity list
always @ (posedge clk) begin
    if (reset) begin
        Data_out <= 'b0;
    end
    else begin
        case (sel)
            5'b00000 : begin
                Data_out[0] <= 'b1;
            end
            5'b00001 : begin
                Data_out[1] <= 'b1;
            end
            5'b00010 : begin
                Data_out[2] <= 'b1;
            end
            5'b00011 : begin
                Data_out[3] <= 'b1;
            end
            5'b00100 : begin
                Data_out[4] <= 'b1;
            end
            5'b00101 : begin
                Data_out[5] <= 'b1;
            end
            5'b00110 : begin
                Data_out[6] <= 'b1; 
            end                  
            5'b00111 : begin     
                Data_out[7] <= 'b1; 
            end                  
            5'b01000 : begin     
                Data_out[8] <= 'b1; 
            end                  
            5'b01001 : begin     
                Data_out[9] <= 'b1; 
            end                  
            5'b01010 : begin     
                Data_out[10] <= 'b1;
            end                  
            5'b01011 : begin     
                Data_out[11] <= 'b1;
            end                  
            5'b01100 : begin     
                Data_out[12] <= 'b1;
            end                  
            5'b01101 : begin     
                Data_out[13] <= 'b1;
            end                  
            5'b01110 : begin     
                Data_out[14] <= 'b1;
            end                  
            5'b01111 : begin     
                Data_out[15] <= 'b1;
            end                  
            5'b10000 : begin     
                Data_out[16] <= 'b1;
            end                  
            5'b10001 : begin     
                Data_out[17] <= 'b1;
            end                  
            5'b10010 : begin     
                Data_out[18] <= 'b1;
            end                  
            5'b10011 : begin     
                Data_out[19] <= 'b1;
            end                  
            5'b10100 : begin     
                Data_out[20] <= 'b1;
            end                  
            5'b10101 : begin     
                Data_out[21] <= 'b1;
            end                  
            5'b10110 : begin     
                Data_out[22] <= 'b1;
            end                  
            5'b10111 : begin     
                Data_out[23] <= 'b1;
            end                  
            5'b11000 : begin     
                Data_out[24] <= 'b1;
            end                  
            5'b11001 : begin     
                Data_out[25] <= 'b1;
            end                  
            5'b11010 : begin     
                Data_out[26] <= 'b1;
            end                  
            5'b11011 : begin     
                Data_out[27] <= 'b1;
            end                  
            5'b11100 : begin     
                Data_out[28] <= 'b1;
            end                  
            5'b11101 : begin     
                Data_out[29] <= 'b1;
            end                  
            5'b11110 : begin     
                Data_out[30] <= 'b1;
            end                  
            5'b11111 : begin     
                Data_out[31] <= 'b1;
            end
            default : begin
                Data_out <= 'b0;
            end
        endcase
    end
    end
    endmodule


