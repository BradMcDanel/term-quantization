`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/10/2019 04:54:47 PM
// Design Name: 
// Module Name: HESE_encoder
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


module HESE_encoder(
    input            clk             ,
    input            reset           ,
    input            input_stream    ,
    input            power_on        ,
    output           output_stream   ,  
    output           sign_stream         
    );

wire [1:0] dummy;    
reg input_stream_reg, input_stream_reg_delay, output_stream_reg, sign_stream_reg;  
wire gated_clk;

assign gated_clk = power_on & clk; 

always @(posedge gated_clk) begin
      if(reset) begin
        {input_stream_reg,input_stream_reg_delay} <= 'b0;
      end else begin
        input_stream_reg <= input_stream;
        input_stream_reg_delay <= input_stream_reg;
      end
    end     

assign dummy[1] = input_stream_reg_delay;
assign dummy[0] = input_stream_reg;
assign output_stream = output_stream_reg;
assign  sign_stream = sign_stream_reg;

always @ (posedge gated_clk) begin
    if(reset) begin
        {output_stream_reg,sign_stream_reg} <= 'b0;
    end
    else begin
        case ({dummy[1],dummy[0]})
            2'b00 : begin
                output_stream_reg <= 'b0;
                sign_stream_reg <='b0;           // zero means positive, and one means negative
            end
            2'b01 : begin
                output_stream_reg <= 'b1;
                sign_stream_reg <='b0;          
            end
            2'b10 : begin
                output_stream_reg <= 'b1;
                sign_stream_reg <='b1;
            end
            2'b11 : begin
                output_stream_reg <= 'b0;
                sign_stream_reg <='b0;          
            end
            default : begin
                output_stream_reg <= 'b0;
                sign_stream_reg <='b0;    
            end
        endcase
    end
end
endmodule
