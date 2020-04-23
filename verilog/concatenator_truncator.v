`timescale 1ns / 1ps


module comparator_truncator#( 
    parameter NUM_TOP_TERMS      =  8                          , // get top NUM_TOP_TERMS terms from consecutive 4 terms
    parameter NUM_TRUNCATED_TERMS     = 4                        // get top NUM_TRUNCATED_TERMS terms from each of the 4 input stream
    ) 
(
    input            clk             ,
    input            sel             ,                           // choose whether use comparator or tuncator
    input            reset           ,
    input            power_on        ,
    input            [4-1:0] input_stream    ,
    input            [4-1:0] input_sign_stream     ,
    output           [4-1:0] output_stream   ,  
    output           [4-1:0] output_sign_stream         
);

reg [2:0] counter0,counter1,counter2,counter3,counter_total;
reg [3:0] output_stream_reg, output_sign_stream_reg;
reg [3:0] output_stream_reg_compare, output_sign_stream_reg_compare;
wire gated_clk;

assign output_sign_stream[3:0] = ({(4){sel}}&output_sign_stream_reg[3:0])|({(4){~sel}}&output_sign_stream_reg_compare[3:0]);
assign output_stream[3:0] = ({(4){sel}}&output_stream_reg[3:0])|({(4){~sel}}&output_stream_reg_compare[3:0]) ;

assign gated_clk = power_on & clk; 

always @(posedge gated_clk) begin
      if(reset) begin
        {counter0,counter1,counter2,counter3,counter_total} <= 'b0;
      end else begin
        counter0 <= counter0 + (input_stream[0] == 1'b1);
        counter1 <= counter1 + (input_stream[1] == 1'b1);
        counter2 <= counter2 + (input_stream[2] == 1'b1);
        counter3 <= counter3 + (input_stream[3] == 1'b1);
        counter_total<= counter_total + (input_stream[0] == 1'b1) + (input_stream[1] == 1'b1) + (input_stream[2] == 1'b1) + (input_stream[3] == 1'b1);
        output_stream_reg[0] <= input_stream[0]&&(counter0<=NUM_TRUNCATED_TERMS);
        output_stream_reg[1] <= input_stream[1]&&(counter1<=NUM_TRUNCATED_TERMS);
        output_stream_reg[2] <= input_stream[2]&&(counter2<=NUM_TRUNCATED_TERMS);
        output_stream_reg[3] <= input_stream[3]&&(counter3<=NUM_TRUNCATED_TERMS);
        output_sign_stream_reg[0] <= input_sign_stream[0]&&(counter0<=NUM_TRUNCATED_TERMS);
        output_sign_stream_reg[1] <= input_sign_stream[1]&&(counter0<=NUM_TRUNCATED_TERMS);
        output_sign_stream_reg[2] <= input_sign_stream[2]&&(counter0<=NUM_TRUNCATED_TERMS);
        output_sign_stream_reg[3] <= input_sign_stream[3]&&(counter0<=NUM_TRUNCATED_TERMS);
        output_stream_reg_compare[0] <= input_sign_stream[0]&&(counter_total<=NUM_TOP_TERMS);
        output_stream_reg_compare[1] <= input_sign_stream[1]&&(counter_total<=NUM_TOP_TERMS);
        output_stream_reg_compare[2] <= input_sign_stream[2]&&(counter_total<=NUM_TOP_TERMS);
        output_stream_reg_compare[3] <= input_sign_stream[3]&&(counter_total<=NUM_TOP_TERMS);
        output_sign_stream_reg_compare[0] <= input_sign_stream[0]&&(counter_total<=NUM_TOP_TERMS);
        output_sign_stream_reg_compare[1] <= input_sign_stream[1]&&(counter_total<=NUM_TOP_TERMS);
        output_sign_stream_reg_compare[2] <= input_sign_stream[2]&&(counter_total<=NUM_TOP_TERMS);
        output_sign_stream_reg_compare[3] <= input_sign_stream[3]&&(counter_total<=NUM_TOP_TERMS);        
      end
end  

    
endmodule
