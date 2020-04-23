// Declaration of serial adder
module bit_serial_adder
(
  input                            clk            ,
  input                            reset          ,
  input                            input_stream   ,
  input                            sign_ctrl      ,
  output       reg                 adder_results 
);

wire another_input;
reg c, b1, b2;   // b1 is the positive 1, b2 is the negative 1
reg [4:0] counter;

assign another_input = b1 & (sign_ctrl) | b2 & (~sign_ctrl);
always@(posedge clk) begin
    if(reset == 1) begin
        counter <= 'b0;
        {b1,b2} <= 'b0;
    end else begin
        counter <= counter + 'b1;
        b1 <= (counter == 5'b11111);
        b2 <= 1'b1;
    end
end
    
always@(posedge clk or posedge reset)  
begin
    if(reset == 1) begin
        adder_results = 0;
    end else begin
        adder_results = input_stream^another_input^c;
        c = (input_stream & another_input) | (another_input & c) | (input_stream & c);
    end
end
endmodule