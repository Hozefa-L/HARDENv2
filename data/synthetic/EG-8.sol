contract BatchedBenefits {
    address payable[] public recipients;
    address public agency;
    function addRecipient(address payable r) external {
        require(msg.sender == agency);
        recipients.push(r);
    }
    function distribute(uint256 amount) external {
        require(msg.sender == agency);
        for (uint i = 0; i < recipients.length; i++) {
            // SWC-113: one revert halts the whole batch
            recipients[i].transfer(amount);
        }
    }
}